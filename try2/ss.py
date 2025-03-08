import random
import numpy as np
from component.beam_search import (
    CoTEnv, SearchTree, LanguageModelCallingFunction,
    RewardModelCallingFunction, RewardModelBaseConfig, ConcatedLMGenResult,
    LMCallingConfig  # 导入 LMCallingConfig
)
# 1. 定义DeepSeek生成模型调用函数
class DeepSeekLM(LanguageModelCallingFunction):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", llm_step_tag="ки\n"):
        super().__init__(llm_step_tag=llm_step_tag)
        self.model_name = model_name
        # 延迟加载模型，避免在初始化时就占用大量内存
        self.tokenizer = None
        self.model = None
        
    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self.model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                print(f"正在加载模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
                print("模型加载完成")
            except Exception as e:
                print(f"模型加载失败: {e}")
                # 如果加载失败，使用模拟数据
                self.tokenizer = None
                self.model = None
                
    def __call__(self, messages: list, config: LMCallingConfig) -> ConcatedLMGenResult:
        try:
            self._ensure_model_loaded()
            
            if self.model is None:
                # 模型加载失败，使用模拟数据
                return self._mock_generation(config)
            
            # 获取配置参数
            n = getattr(config, 'n', 2)  # 默认生成2个结果
            temperature = getattr(config.generation_config, 'temperature', 0.7) if hasattr(config, 'generation_config') else 0.7
            max_length = getattr(config, 'max_length', 100)
            
            # 准备输入
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict) and 'content' in messages[0]:
                    # 处理消息格式的输入
                    prompt = messages[0]['content']
                else:
                    # 处理其他格式的输入
                    prompt = str(messages)
            else:
                prompt = str(messages)
            
            # 生成文本
            results = []
            logps = []
            
            for _ in range(n):
                # 使用AutoTokenizer和AutoModelForCausalLM直接生成文本
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                
                # 生成文本
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # 解码生成的文本
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 移除原始提示部分，只保留生成的内容
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                
                # 添加步骤前缀
                if not text.startswith("Step"):
                    text = f"Step 1: {text}"
                
                results.append(text)
                # 模拟对数概率（实际模型可能提供这个值）
                logps.append(-0.1 * random.random())
            
            # 计算token数量（简化处理）
            token_counts = [len(text.split()) for text in results]
            
            return ConcatedLMGenResult(
                text=results,
                prompt_tokens=[len(prompt.split())] * n,
                num_tokens=token_counts,
                cumulative_logprob=logps,
                logp_avg_by_len=logps,
                finish_reason=["stop"] * n
            )
            
        except Exception as e:
            print(f"生成过程中出错: {e}")
            return self._mock_generation(config)
    
    def _mock_generation(self, config):
        """当模型加载失败或生成出错时使用模拟数据"""
        n = getattr(config, 'n', 2)  # 默认生成2个结果
        texts = [f"Step 1: Let's solve this step by step...\n" for _ in range(n)]
        logps = [-0.1 * i for i in range(len(texts))]
        token_counts = [20] * len(texts)
        
        return ConcatedLMGenResult(
            text=texts,
            prompt_tokens=[10] * len(texts),
            num_tokens=token_counts,
            cumulative_logprob=logps,
            logp_avg_by_len=logps,
            finish_reason=["stop"] * len(texts)
        )

# 2. 定义随机PRM奖励模型
class RandomRM(RewardModelCallingFunction):
    def __init__(self):
        # 正确初始化RewardModelBaseConfig对象
        config = RewardModelBaseConfig()
        config.prm_step_tag = "ки\n"  # 设置步骤标签
        config.format_str = "{question}\n{answer}"  # 设置格式字符串
        config.rm_serve_type = "random"  # 设置奖励模型服务类型
        config.step_tag_id = 0  # 设置步骤标签ID
        config.returned_token_ids = []  # 设置返回的token IDs
        super().__init__(config)
        
    def __call__(self, question_answer_pairs, model_names):
        # 生成随机PRM奖励值
        if isinstance(question_answer_pairs, list):
            return [[random.random() for _ in range(3)] for _ in question_answer_pairs]  # 假设每个步骤有3个奖励值
        else:
            return [[random.random()]]

# 3. 配置环境参数
config = {
    "max_actions": 5,
    "beam_size": 2,
    "max_length": 3,
    "generation_config": {"temperature": 0.7},
    "direct_io": 0
}

# 4. 初始化数学问题和模型
math_problems = [{"question": "If x + 5 = 12, what is x?"}]
deepseek_lm = DeepSeekLM()
random_rm = RandomRM()  # 修改这里，不再传入RewardModelBaseConfig

# 5. 创建CoT环境
cot_env = CoTEnv(
    config=config,
    math_problems=math_problems,
    llm_gen_fns=[deepseek_lm],
    rm_call=random_rm,
    task_desc_str="Solve the equation step by step:",
    cot_example_str="Example: 3 + 4 = 7",
    problem_format_str="Problem: {question}\nSolution:",
    sep="\n"
)

# 6. 初始化搜索树并执行波束搜索
search_tree = SearchTree(cfg={
    "num_simulations": 2,
    "pb_c_base": 19652,
    "root_dirichlet_alpha": 0.3,
    "model_names": ["deepseek"]
})

results = search_tree.beam_search(
    simulate_env=cot_env,
    beam_size=2,
    max_step=3,
    reward_model_fn=lambda x: [[random.random() for _ in range(3)] for _ in range(len(simulate_env.legal_actions))]  # 最终状态随机奖励，返回列表而非单个值
)

# 7. 输出结果
for result in results:
    print(f"Answer: {result['text']}")
    print(f"Value: {result['value']:.2f}")
    print("="*50)