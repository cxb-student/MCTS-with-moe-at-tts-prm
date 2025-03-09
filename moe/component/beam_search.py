import abc
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import torch.distributed as dist
import heapq
from loguru import logger
import copy
import json
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
class BaseEnv(abc.ABC):
    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action, update_legal_action=True):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def legal_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    def build_query_str(
        cot_task_desc: Optional[str],
        cot_examples: Optional[str],
        problem_format_str: str,
        problem_input: str,
        is_few_shot: bool = False,
        model_names = [],
    ):
        """
        a wrap function that wrap the problem text with certrain format
        e.g. prompt_str = "Input: " + join_numbers(" ", xs) + "\nSteps:\n"
        # >>> query_str = Game24Env.build_query_str("1 1 1 1")
        # >>> print(query_str)
        # >>> Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
        # Input: 1 1 1 1
        Steps:
        """

        messages = []
        problem_format_str = problem_format_str.format(question=problem_input)
        messages.append({"role": "user", "content": problem_format_str + '\n' + cot_task_desc})
        return messages

    @staticmethod
    def build_response_str(answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool):
        raise NotImplementedError
@dataclass
class LMCallingConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 for vllm by default
    max_new_tokens: int = 512
    stop_token_ids: Optional[List[int]] = None
    stop_str: Optional[Union[str, List[str]]] = None
    include_stop_str_in_output: bool = False
    first_generation: bool = False

@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)
class NoLegalActionException(Exception):
    pass
def print_with_rank(message):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print("[{}/{}]: {}".format(rank, world_size, message), flush=True)
    else:
        print(message, flush=True)

class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""
    def _is_correct(self, completion) -> bool:
        # 使用奖励模型评分判断动作是否正确
        # 获取当前状态文本
        current_state = self.get_state(model_name='raw')
        # 使用rm_call计算奖励
        reward = self.rm_call(current_state, self.model_names)
        if isinstance(reward, list) and len(reward) > 0 and isinstance(reward[0], list) and len(reward[0]) > 0:
            # 如果奖励值大于0.5，则认为动作是正确的
            return reward[0][0] > 0.5
        return False
    def get_reward(self):
        """使用rm_call函数计算当前状态的奖励值"""
        if len(self.action_history) > 0:
            current_state = self.get_state(model_name='raw')
            reward = self.rm_call(current_state, self.model_names)
            if isinstance(reward, list) and len(reward) > 0 and isinstance(reward[0], list) and len(reward[0]) > 0:
                return reward[0][0]
            return 0.0
        return 0.0  # 初始状态返回0奖励

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fns,
        rm_call,
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
        sep=None,
        model_names=[],
        update_legal_action=True,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.llm_gen_fns = llm_gen_fns
        self.rm_call = rm_call
        self.action_history = None
        self.reward_history = None
        self.token_history = None
        self.prob_history = None
        self.model_history = None
        self.math_problem = None
        self._legal_actions = None
        self._stop_str = None  # 初始化_stop_str属性
        self.is_few_shot = config.get("is_few_shot", False)
        self.add_step_prompt = config.get("add_step_prompt", False)
        self.direct_io = config.get("direct_io", 0)
        self.double_line_break = config.get("double_line_break", 0)
        # self.prm_step_tag = rm_call.prm_step_tag  # "ки\n"
        self.prm_step_tag = "ки\n"
        self.sep = sep
        self.model_names = model_names

        if config.get("cot_prompt", ""):
            task_desc_str = config["cot_prompt"]
        self._task_desc_str = task_desc_str
        self._cot_example_str = cot_example_str
        self._problem_format_str = problem_format_str

        prefixes = []
        if self._task_desc_str is not None:
            prefixes.append(self._task_desc_str)
        if self.is_few_shot:
            prefixes.append(self._cot_example_str)
        if len(prefixes) > 0:
            self.task_prefix = "\n".join(prefixes)
        else:
            self.task_prefix = None

        if reset:
            self.reset(update_legal_action=update_legal_action)

    def reset(self, update_legal_action=True):
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = []
        self.reward_history = []
        self.token_history = []
        self.prob_history = []
        self.model_history = []
        self._init_query = self.build_query_str(
            cot_examples=self._cot_example_str,
            cot_task_desc=self._task_desc_str,
            problem_format_str=self._problem_format_str,
            problem_input=self.math_problem["Q"],
            is_few_shot=self.is_few_shot,
            model_names=self.model_names,
        )
        if update_legal_action:
            cnt = 0
            max_try = 1
            while cnt < max_try:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = self.update_legal_actions(initial=True)
                    break
                except Exception as e:
                    if cnt == max_try:
                        self._legal_actions, api_completion_token = self.update_legal_actions(initial=True, force_update=True)
                        print("Force update legal actions:", self._legal_actions)
        else:
            api_completion_token = 0
        info = {"api_completion_token": api_completion_token}
        return self.get_state(model_name='raw'), info

    def step(self, action, update_legal_action=True, model_name="", custom_n=0, reward=0.0, num_token=0, prob=0.0):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.token_history.append(num_token)
        self.prob_history.append(prob)
        if model_name:
            self.model_history.append(model_name)
        state = self.get_state(model_name=model_name)
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()

        if not (terminated or truncated) and update_legal_action:  # update legal actions
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = self.update_legal_actions(custom_n=custom_n)
                    info["api_completion_token"] = api_completion_token
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["winner"] = 2
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
            info["api_completion_token"] = 0
        return state, reward, terminated, truncated, info

    def get_state(self, model_name='other', add_step_prompt=False):
        messages = copy.deepcopy(self._init_query)
        messages.append({"role": "assistant", "content": "".join(self.action_history)})

        if add_step_prompt and self.direct_io != 2:
            if 'llama-3' in self.model_names[0].lower():  # TODO: Check llama
                sep = "## Step"
            else:
                sep = "Step"
            if not self.double_line_break:  # TODO: Check double
                messages[-1]["content"] += f"{sep} {len(self.action_history) + 1}: "
        if model_name == 'raw':
            ret = ""
            for idx, mess in enumerate(messages):
                ret += f'{mess["role"]}: {mess["content"]}'
                if idx < len(messages) - 1:
                    ret += '\n'
            return ret
        return messages

    def post_process_act(self, action: str):
        # This step may change the token count
        return action

    def update_legal_actions(self, initial=False, force_update=False, custom_n=0):
        if len(self.llm_gen_fns) == 1:
            if initial:
                n = self.config["max_actions"]
            elif custom_n:
                n = custom_n
            else:
                n = self.config["max_actions"] // self.config["beam_size"]
            if self.direct_io:
                stop_str, include_stop_str_in_output = None, False
            else:
                stop_str, include_stop_str_in_output = self.sep, True
            first_generation = len(self.action_history) == 0
            messages = self.get_state(self.llm_gen_fns[0].model_name, add_step_prompt=self.add_step_prompt)
            # Create LMCallingConfig instance
            config = LMCallingConfig(
                n=n,
                stop_str=stop_str,  # '\n\n' for Qwen-2.5-Math-1.5B-Instruct
                include_stop_str_in_output=include_stop_str_in_output,
                first_generation=first_generation,
            )
            
            # Add generation_config parameters
            for key, value in self.config["generation_config"].items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
            result: ConcatedLMGenResult = self.llm_gen_fns[0](
                messages=messages,
                config=config,
            )
            texts = result.text  # [text1, text2]
            logps_avg_by_len = result.logp_avg_by_len  # [-0.10557132510029904, -0.23053854329903292]
            token_len = result.num_tokens  # [212, 192]
            temp_model_names = [self.llm_gen_fns[0].model_name] * len(texts)
            temp_model_ids = [0] * len(texts)
            finish_reason_list = []
            if isinstance(result.finish_reason, list):
                finish_reason_list.extend(result.finish_reason)
            else:
                raise ValueError("finish_reason should be a list")
        else:
            raise NotImplementedError

        text_list, prob_list, num_token_list = [], [], []
        model_names, model_ids = [], []
        next_state_terminated = {}
        raw_text_list = []

        for i in range(len(texts)):
            if self.direct_io:
                terminated = True
            else:
                if isinstance(self.sep, str):
                    terminated = not texts[i].endswith(self.sep)
                elif isinstance(self.sep, list):
                    terminated = True
                    for sep in self.sep:
                        if texts[i].endswith(sep):
                            terminated = False
                            break
            processed_act = self.post_process_act(texts[i])
            finish_reason = finish_reason_list[i]
            if not self.double_line_break:
                temp_act = processed_act.replace("## Step ", "Step ")
                is_double_line_break = temp_act.endswith("\n\n") and temp_act.startswith("Step ") and (len(temp_act) == len("Step 1: \n\n") or len(temp_act) == len("Step 10: \n\n"))
                if is_double_line_break:
                    finish_reason = "length"
            if len(processed_act) > 0 and processed_act not in text_list and finish_reason == "stop":
                # only stop is valid, otherwise the output action is truncated actually
                text_list.append(processed_act)
                raw_text_list.append(texts[i])
                prob_list.append(logps_avg_by_len[i])
                num_token_list.append(token_len[i])
                next_state_terminated[processed_act] = terminated
                model_names.append(temp_model_names[i])
                model_ids.append(temp_model_ids[i])
            elif force_update or self.direct_io:
                text_list.append(processed_act)
                raw_text_list.append(texts[i])
                prob_list.append(logps_avg_by_len[i])
                num_token_list.append(token_len[i])
                next_state_terminated[processed_act] = terminated
                model_names.append(temp_model_names[i])
                model_ids.append(temp_model_ids[i])

        if len(prob_list) == 0:
            print_with_rank("state: {}".format(self.get_state(model_name='raw')))
            if len(self.llm_gen_fns) == 1:
                print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = np.exp(prob_list)
        prob_list = list(prob_list)
        # prob_list = np.array(prob_list)
        # prob_list = prob_list / np.sum(prob_list)  # normalize probability

        _legal_actions = [{
            "action": action,
            "prob": prob,
            "num_token": n_token,
            "finish_reason": finish_reason,
            "model_name": model_name,
            "model_id": model_id,
            "messages": messages,
            "stop_str": stop_str,
            "raw_action": raw_action,
        } for action, prob, n_token, finish_reason, model_name, model_id, raw_action in zip(text_list, prob_list, num_token_list,
            finish_reason_list, model_names, model_ids, raw_text_list)]

        if len(self.llm_gen_fns) == 1:
            completion_tokens = result.completion_tokens
        self._next_state_terminated = next_state_terminated

        return _legal_actions, completion_tokens

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def query(self):
        return self._init_query

    @property
    def question(self) -> str:
        return self.math_problem["question"]

    @property
    def answer(self):
        if len(self.action_history) == 0:
            return ""
        elif self.direct_io == 2:
            assert len(self.action_history) == 1
            return self.action_history[0]
        elif self.direct_io == 1:
            assert len(self.action_history) == 1
            steps = self.action_history[0].split("\n\n")
            answer = ""
            for step in steps:
                if step.strip() == "":
                    continue
                answer += step.strip() + f" {self.prm_step_tag}"
            return answer
        else:
            answer = ""
            for action in self.action_history:
                answer += action.strip() + f" {self.prm_step_tag}"
            return answer

    def check_stop_by_answer(self):
        if isinstance(self._stop_str, str) and self._stop_str in self.action_history[-1]:
            terminated = True
        elif isinstance(self._stop_str, list):
            terminated = True
            for stop_str in self._stop_str:
                if stop_str not in self.action_history[-1]:
                    terminated = False
        return terminated

    def check_stop_by_sep(self):
        if isinstance(self.sep, str):
            return self.sep not in self.action_history[-1]
        elif isinstance(self.sep, list):
            for sep in self.sep:
                if sep in self.action_history[-1]:
                    return False
        return False

    def get_done_and_info(self):
        info = {"winner": 0}
        # done when reaches maximum length or LLM generates stop words
        if self._stop_str is not None and self.check_stop_by_answer():
            terminated = True
        elif self._next_state_terminated[self.action_history[-1]]:
            terminated = True
        else:
            terminated = self.check_stop_by_sep()

        if self.config["max_length"] > 1:
            truncated = len(self.action_history) >= self.config["max_length"]
            assert len(self.action_history) <= self.config["max_length"]
        else:
            truncated = False
        if terminated or truncated:
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1
            else:
                info["winner"] = 2
            return terminated, truncated, info
        return terminated, truncated, info

    def copy(self):
        env = self.__class__(
            self.config,
            self.math_problems,
            self.llm_gen_fns,
            self.rm_call,
            self._task_desc_str,
            self._cot_example_str,
            self._problem_format_str,
            reset=False,
        )
        env.math_problem = copy.deepcopy(self.math_problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        env.reward_history = copy.deepcopy(self.reward_history)
        env.token_history = copy.deepcopy(self.token_history)
        env.prob_history = copy.deepcopy(self.prob_history)
        env.model_history = copy.deepcopy(self.model_history)
        env._init_query = copy.deepcopy(self._init_query)
        env._next_state_terminated = copy.deepcopy(self._next_state_terminated)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions
class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0, parent_value: float = 0.0) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p

        self._initial_value = initial_value
        self._parent_value = parent_value
        self._terminated = False

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._initial_value
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Update the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> bool:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets

    def __str__(self) -> str:
        if self.is_root():
            return "root"
        else:
            return "child: value: {:.3f}, prior: {:.3f}".format(self.last_action, self.value, self.prior_p)
class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        parent_value: float = 0.0,
        num_generated_token: Optional[int] = None,
        model_name: str = "",
    ) -> None:
        super().__init__(parent, prior_p, initial_value, parent_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

        self.model_name = model_name

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
            info_dict["prm_value"] = self.prm_value
        else:
            info_dict["text_state"] = self.text_state
        return info_dict

    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.text_state)
        else:
            return "action: {}, value: {:.3f}, prior: {:.3f}".format(self.last_action, self.value, self.prior_p)
class LanguageModelCallingFunction:

    def __init__(self, llm_step_tag: str = None):
        self.llm_step_tag = llm_step_tag

    def __call__(self, messages: List, config: LMCallingConfig) -> ConcatedLMGenResult:
        raise NotImplementedError
class RewardModelBaseConfig:
    prm_step_tag: str
    format_str: str  # a format string that takes in question and answer need to have {question} and {answer} in the string

    rm_serve_type: str
    step_tag_id: int
    returned_token_ids: List[int]
class RewardModelCallingFunction:

    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.prm_step_tag = config.prm_step_tag
        self.format_str = config.format_str

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    def replace_step_tag(self, answer: str):
        if self.prm_step_tag not in answer:
            answer += f" {self.prm_step_tag}"
        splits = answer.split(f" {self.prm_step_tag}")
        splits = [s.strip() for s in splits]
        response = f" {self.prm_step_tag}".join([s for s in splits if s != ""])
        response += f" {self.prm_step_tag}"
        return response

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


class SearchTree:
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._num_simulations = self._cfg.get("num_simulations", 20)
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)
        self._root_dirichlet_alpha = self._cfg.get("root_dirichlet_alpha", 0.3)
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)
        self.root = None
        self.answers = set()
        self.wrong_answers = set()
        self.visited_paths = None
        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        self.mask_non_terminal_node_value = self._cfg.get("mask_non_terminal_node_value", False)
        self._init_critic_value = self._cfg.get("init_critic_value", True)
        self._completion_tokens = 0
        self.model_names = self._cfg.get("model_names", [])
        self.direct_io = self._cfg.get("direct_io", 0)
        self.max_actions = self._cfg.get("max_actions", 0)

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)

    def beam_search(
            self,
            simulate_env: "CoTEnv",
            beam_size: int,
            max_step: int,
            reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        修改后的 beam_search 输出 5 个候选答案，每个候选包含：
          - 最终生成文本（text）
          - 累计 log 概率（total_log_prob）
          - 轨迹概率（trajectory_probability，= exp(total_log_prob)）
          - 每一步的 reward_history、token_history、prob_history、model_history、logits_history 等
          - 总奖励（total_reward，= sum(reward_history)）
        便于后续 GRPO 损失的计算和训练生成模型与 PRM。
        """
        if max_step == 1:
            assert self.direct_io, "当 max_step == 1 时应使用 direct_io 模式。"

        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]

        # 初始化 root 节点，累计 log 概率从 0 开始
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state(model_name='raw'))
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        # 每个候选轨迹记录为一个 tuple:
        # (neg_total, neg_value, neg_parent_value, node, env, total_log_prob)
        # 其中 total_log_prob 表示从根节点到当前节点累计的 log 概率
        end_nodes = []
        top_k_nodes = [(-root._initial_value, -root._initial_value, -root._parent_value,
                        root, simulate_env.copy(), 0.0)]
        k = beam_size

        for step in range(max_step + 1):
            next_top_k = []
            for neg_total, neg_value, neg_parent_value, node, env, total_log_prob in top_k_nodes:
                if node.terminated:
                    end_nodes.append((neg_total, neg_value, neg_parent_value, node, env, total_log_prob))
                    continue
                # 如果当前节点还没有扩展，则先扩展（确保 children 非空）
                if not node.children:
                    self._expand_leaf_node(node, env, reward_model_fn)
                if not node.children:
                    node.set_as_terminate_node()
                    end_nodes.append((neg_total, neg_value, neg_parent_value, node, env, total_log_prob))
                    continue

                # 针对 direct_io 与非 direct_io 模式分别处理
                if self.direct_io:
                    for child_idx, child in node.children.items():
                        # 这里 child.prior_p 为生成时的概率，计算 log 后累加
                        new_log_prob = np.log(child.prior_p + 1e-8)
                        new_total_log_prob = total_log_prob + new_log_prob
                        new_env = env.copy()
                        _, _, terminated, truncated, info = new_env.step(
                            child.last_action,
                            update_legal_action=(self.direct_io == 0),
                            model_name=child.model_name,
                            reward=child._initial_value,
                            num_token=child.num_generated_token,
                            prob=child.prior_p,
                        )
                        api_call_completion_tokens += info["api_completion_token"]
                        new_item = (-child._initial_value, -child._initial_value, -child._parent_value,
                                    child, new_env, new_total_log_prob)
                        next_top_k.append(new_item)
                else:
                    for action, child in node.children.items():
                        new_log_prob = np.log(child.prior_p + 1e-8)
                        new_total_log_prob = total_log_prob + new_log_prob
                        new_env = env.copy()
                        _, _, terminated, truncated, info = new_env.step(
                            child.last_action,
                            update_legal_action=True,
                            model_name=child.model_name,
                            reward=child._initial_value,
                            num_token=child.num_generated_token,
                            prob=child.prior_p,
                        )
                        api_call_completion_tokens += info["api_completion_token"]
                        new_item = (-child._initial_value, -child._initial_value, -child._parent_value,
                                    child, new_env, new_total_log_prob)
                        next_top_k.append(new_item)
            if len(end_nodes) >= beam_size:
                break
            if next_top_k:
                top_k_nodes = heapq.nsmallest(beam_size, next_top_k)
            else:
                break

        # 合并所有候选：已终止的轨迹和当前未终止的轨迹
        all_candidates = end_nodes + top_k_nodes
        traj_list = []
        for i, (neg_total, neg_value, neg_parent_value, node, env, total_log_prob) in enumerate(all_candidates):
            # 计算轨迹概率（exp(total_log_prob)）和总奖励（reward_history 累加）
            trajectory_probability = float(np.exp(total_log_prob))
            total_reward = float(np.mean(env.reward_history)) if env.reward_history else 0.0
            traj_list.append({
                "path_idx": i,
                "text": env.answer,
                "value": -neg_value,
                "parent_value": -neg_parent_value,
                "q_plus_a": -neg_total,
                "total_log_prob": float(total_log_prob),
                "trajectory_probability": trajectory_probability,
                "total_reward": total_reward,
                "api_completion_tokens": 0,
                "tree_completion_tokens": self._completion_tokens,
                "reward_history": env.reward_history,
                "token_history": env.token_history,
                "prob_history": env.prob_history,
                "model_history": env.model_history,
                "logits_history": env.logits_history,
            })
        if traj_list:
            traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens
        final_candidates = sorted(traj_list, key=lambda x: x["total_log_prob"], reverse=True)[:5]
        return final_candidates

    def _select_child(self, node: LanguageNode, simulate_env: CoTEnv) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """

        action = None
        child = None
        best_score = -9999999

        for action_tmp, child_tmp in node.children.items():
            ucb_score = self._ucb_score(node, child_tmp)
            score = ucb_score
            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child

    def _select_by_prior(self, node: Node, simulate_env: CoTEnv):
        data_tmp = [(x_action, x_node.prior_p) for x_action, x_node in node.children.items()]
        action_list, prior_list = list(zip(*data_tmp))
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action]

        return chosen_action, chosen_node

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: CoTEnv,
        rm_call: Optional[Callable] = None,
    ) -> float:
        """
        Overview:
            expand the node with the rm_call.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - rm_call (:obj:`Function`): the Callable to compute the state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        """
        action_probs_dict, leaf_value = rm_call(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        """

        text_state = simulate_env.get_state(model_name='raw')
        if not self._init_critic_value:
            leaf_value = rm_call(text_state)
        else:
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0
            if self.direct_io:
                prms = [[0.0] for _ in simulate_env.legal_actions]
            else:
                prm_inputs = [(simulate_env.question, simulate_env.answer + x["action"]) for x in simulate_env.legal_actions]
                for i in range(2):
                    try:
                        prms = rm_call(prm_inputs)
                        break
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        # prms = [[0.0] for _ in simulate_env.legal_actions]
            child_values = []
            for act, rs in zip(simulate_env.legal_actions, prms):
                if len(simulate_env.action_history) + 1 != len(rs):
                    logger.warning(f"PRM value length not match with action history. len(prm)={len(rs)}, "
                                   f"len(action_history)={len(simulate_env.action_history)}\ns:\n{text_state}\na:\n{act}\nrs:{rs}")
                    try:
                        prm = rm_call([(simulate_env.question, simulate_env.answer + x["action"]) for x in [act]], verbose=True, legal_action=[act])
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                    child_values.append(0.0)
                elif len(rs) == 0:
                    logger.warning(f"Empty PRM value for: \nState: \n{text_state} \naction: \n{act}, will be set to 0.0")
                    child_values.append(0.0)
                else:
                    # prm-last
                    child_values.append(rs[-1])  # PRM get last r as single reward, [0.9783847332000732, 0.9621075391769409]
                    # # prm-min
                    # child_values.append(min(rs))
                    # # prob-prm
                    # child_values.append(act['prob'])

        assert len(node.children) == 0
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]
            model_name = action_dict["model_name"]

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # XXX(ziyu): consider turn off this branch, i.e. always assume
                #  `self._init_critic=True`, since with LLM
                child_value = 0.0

            if self.direct_io:
                node.children[i] = LanguageNode(
                    parent=node,
                    prior_p=prob,
                    # prm_value=prm_value,
                    text_state=text_state,
                    last_action=action,
                    initial_value=child_value,
                    parent_value=leaf_value,
                    num_generated_token=action_dict["num_token"],
                    model_name=model_name,
                )
            else:
                node.children[action] = LanguageNode(
                    parent=node,
                    prior_p=prob,
                    # prm_value=prm_value,
                    text_state=text_state,
                    last_action=action,
                    initial_value=child_value,
                    parent_value=leaf_value,
                    num_generated_token=action_dict["num_token"],
                    model_name=model_name,
                )
            # set terminal node here
            if simulate_env._next_state_terminated[action]:
                if self.direct_io:
                    node.children[i].set_as_terminate_node()
                else:
                    node.children[action].set_as_terminate_node()
        if len(node.children) == 0:
            print_rank_0("Prune all current children at node {}".format(node.last_action))

        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(c.num_generated_token for c in node.children.values())
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

        return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        pb_c = (math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init)
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value

        return prior_score + value_score
        # return value_score

    def reset_prior(self, node: Node) -> None:
        """
        Overview:
            Reset prior probability
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        for a in node.children.keys():
            node.children[a].prior_p = node.children[a].prior_p_ori

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            node_info = tree_dict["info"]
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
            )

            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        root_node = build_tree(tree_dict=tree_json)

        obj = cls(cfg)
        obj.root = root_node
        return obj
    def draw_tree(self):
        # Not tested yet
        root = self.root
        assert root, 'Root node is None'
        def draw_node(node, depth):
            print('|' + '-' * depth + str(node))
            for child in node.children.values():
                draw_node(child, depth + 1)
        print(f"\n---------Expanded Tree---------")
        draw_node(self.root, 0)
