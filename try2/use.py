import torch
import numpy as np
import random
from component.beam_search import (
    CoTEnv, SearchTree, LanguageModelCallingFunction, 
    RewardModelCallingFunction, RewardModelBaseConfig, ConcatedLMGenResult,
    LMCallingConfig  # 导入 LMCallingConfig
)
from typing import List, Dict, Optional, Tuple, Union, Callable

# 1. 创建一个模拟的 DeepSeek 模型调用函数
class DeepSeekModelCaller(LanguageModelCallingFunction):
    def __init__(self, model_name="deepseek"):
        super().__init__(llm_step_tag="ки\n")
        self.model_name = model_name
    
    def __call__(self, messages: List, config):
        # 处理 config 参数，确保它是 LMCallingConfig 实例
        if not isinstance(config, LMCallingConfig):
            # 如果不是 LMCallingConfig 实例，创建一个新的实例并复制属性
            lm_config = LMCallingConfig()
            for key, value in vars(config).items():
                if hasattr(lm_config, key):
                    setattr(lm_config, key, value)
            config = lm_config
            
        # 模拟 DeepSeek 模型生成结果
        n = config.n
        texts = []
        logp_avg_by_len = []
        num_tokens = []
        finish_reason = []
        
        for i in range(n):
            # 生成随机文本，模拟模型输出
            token_length = random.randint(50, 200)
            text = f"Step {i+1}: 这是一个模拟的 DeepSeek 模型输出，解决问题的步骤 {i+1}。\n\n"
            
            texts.append(text)
            logp_avg_by_len.append(-random.random())  # 随机对数概率
            num_tokens.append(token_length)
            finish_reason.append("stop")
        
        return ConcatedLMGenResult(
            text=texts,
            prompt_tokens=[len(str(messages))] * n,
            num_tokens=num_tokens,
            cumulative_logprob=[l * t for l, t in zip(logp_avg_by_len, num_tokens)],
            logp_avg_by_len=logp_avg_by_len,
            finish_reason=finish_reason
        )

# 其余代码保持不变...
# 2. 创建一个随机的奖励模型调用函数
class RandomRewardModelCaller(RewardModelCallingFunction):
    def __init__(self):
        config = RewardModelBaseConfig()
        config.prm_step_tag = "ки\n"
        config.format_str = "{question}\n{answer}"
        config.rm_serve_type = "random"
        config.step_tag_id = 0
        config.returned_token_ids = []
        super().__init__(config)
    
    def __call__(self, question_answer_pairs, model_names=None):
        # 为每个问答对生成随机奖励值
        results = []
        for _ in question_answer_pairs:
            # 为每个步骤生成一个随机奖励值
            step_rewards = [random.random() for _ in range(random.randint(1, 3))]
            results.append(step_rewards)
        return results

# 3. 创建一个简单的数学问题
math_problems = [
    {
        "question": "计算 125 + 37 的结果是多少?",
        "answer": "162"
    }
]

# 4. 配置环境
config = {
    "max_actions": 5,
    "max_length": 3,
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "beam_size": 3,
    "num_simulations": 10,
    "pb_c_base": 19652,
    "pb_c_init": 1.25,
    "root_dirichlet_alpha": 0.3,
    "root_noise_weight": 0.25,
    "direct_io": 0,
    "model_names": ["deepseek"],
    "is_few_shot": False,
    "add_step_prompt": True,
}

# 5. 实例化模型调用函数
deepseek_model = DeepSeekModelCaller()
random_reward_model = RandomRewardModelCaller()

# 6. 创建环境
# ... 前面的代码保持不变 ...
class CustomCoTEnv(CoTEnv):
    def get_reward(self):
        """实现奖励计算方法"""
        if len(self.reward_history) > 0:
            return self.reward_history[-1]
        return 0.0
    
    def _is_correct(self, completion):
        """判断答案是否正确"""
        return "162" in completion
# 6. 创建环境
task_desc = "请一步一步地解决这个数学问题。"
cot_example = "问题: 计算 5 + 7 的结果是多少?\n步骤 1: 将 5 和 7 相加。\n步骤 2: 5 + 7 = 12\n答案: 12"
problem_format = "{question}"

env = CustomCoTEnv(
    config=config,
    math_problems=math_problems,
    llm_gen_fns=[deepseek_model],
    rm_call=random_reward_model,
    task_desc_str=task_desc,
    cot_example_str=cot_example,
    problem_format_str=problem_format,
    sep="\n\n",
    model_names=config["model_names"],
    update_legal_action=False
)

# 修补 CoTEnv 的 update_legal_actions 方法
original_update_legal_actions = env.update_legal_actions

def patched_update_legal_actions(self, initial=False, force_update=False, custom_n=0):
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
        
        # 创建 LMCallingConfig 实例并设置属性
        lm_config = LMCallingConfig()
        lm_config.n = n
        lm_config.stop_str = stop_str
        lm_config.include_stop_str_in_output = include_stop_str_in_output
        lm_config.first_generation = first_generation
        
        # 添加 generation_config 中的配置
        for key, value in self.config["generation_config"].items():
            setattr(lm_config, key, value)
            
        result = self.llm_gen_fns[0](
            messages=messages,
            config=lm_config,
        )
        
        # 处理结果
        texts = result.text
        logps_avg_by_len = result.logp_avg_by_len
        token_len = result.num_tokens
        temp_model_names = [self.llm_gen_fns[0].model_name] * len(texts)
        temp_model_ids = [0] * len(texts)
        finish_reason_list = []
        if isinstance(result.finish_reason, list):
            finish_reason_list.extend(result.finish_reason)
        else:
            raise ValueError("finish_reason should be a list")
            
        # 继续原始方法的其余部分
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
            from component.beam_search import print_with_rank, NoLegalActionException
            print_with_rank("state: {}".format(self.get_state(model_name='raw')))
            if len(self.llm_gen_fns) == 1:
                print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = np.exp(prob_list)
        prob_list = list(prob_list)

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

        completion_tokens = result.completion_tokens
        self._next_state_terminated = next_state_terminated

        return _legal_actions, completion_tokens
    else:
        raise NotImplementedError

# 应用猴子补丁
env.update_legal_actions = lambda initial=False, force_update=False, custom_n=0: patched_update_legal_actions(env, initial, force_update, custom_n)

# 现在手动调用 reset 来初始化环境
_, info = env.reset(update_legal_action=True)

# 7. 创建搜索树
search_tree = SearchTree(config)

# 8. 执行 Beam Search
def main():
    print("开始执行 Beam Search...")
    trajectories = search_tree.beam_search(
        simulate_env=env,
        beam_size=config["beam_size"],
        max_step=config["max_length"],
        reward_model_fn=random_reward_model
    )
    
    print(f"\n找到 {len(trajectories)} 条路径:")
    for i, traj in enumerate(trajectories):
        print(f"\n路径 {i+1}:")
        print(f"文本: {traj['text']}")
        print(f"值: {traj['value']:.4f}")
        print(f"父节点值: {traj['parent_value']:.4f}")
        print(f"Q+A 值: {traj['q_plus_a']:.4f}")
        print(f"API 调用生成的 token 数: {traj['api_completion_tokens']}")
        print(f"树搜索生成的 token 数: {traj['tree_completion_tokens']}")
        print(f"奖励历史: {traj['reward_history']}")
        print(f"Token 历史: {traj['token_history']}")
        print(f"概率历史: {traj['prob_history']}")
        print(f"模型历史: {traj['model_history']}")

if __name__ == "__main__":
    main()