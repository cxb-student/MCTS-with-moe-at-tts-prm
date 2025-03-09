import random
import torch
import sys
import os
# 将项目根目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（inference）
root_dir = os.path.dirname(current_dir)                   # 根目录（moe）
sys.path.append(root_dir)
from component.beam_search import CoTEnv, SearchTree, LanguageModelCallingFunction,RewardModelCallingFunction, RewardModelBaseConfig, ConcatedLMGenResult,LMCallingConfig  # 导入 LMCallingConfig
from component.lora_config import get_lora_config, apply_lora_to_model

# 1. 定义DeepSeek生成模型调用函数
class DeepSeekLM(LanguageModelCallingFunction):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", llm_step_tag="ки\n"):
        super().__init__(llm_step_tag=llm_step_tag)
        self.model_name = model_name
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
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                
                # 生成文本
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
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

# 2. 定义Qwen PRM奖励模型
class QwenPRM(RewardModelCallingFunction):
    def __init__(self):
        # 初始化RewardModelBaseConfig对象
        config = RewardModelBaseConfig()
        config.prm_step_tag = "<extra_0>"  # Qwen特殊标记
        config.format_str = "{question}\n{answer}"  # 设置格式字符串
        config.rm_serve_type = "qwen"  # 设置奖励模型服务类型
        super().__init__(config)
        
        # 加载Qwen模型
        self.model_name = "Qwen/Qwen2.5-Math-7B-PRM800K"
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
            
            # 应用LoRA配置

            from peft import TaskType
            lora_config = get_lora_config(r=4, lora_alpha=16, task_type=TaskType.CAUSAL_LM)
            self.model = apply_lora_to_model(base_model, lora_config)
            print("Qwen PRM模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def __call__(self, question_answer_pairs, model_names):
        if self.model is None:
            # 如果模型加载失败，返回随机值
            return [[random.random() for _ in range(3)] for _ in question_answer_pairs] if isinstance(question_answer_pairs, list) else [[random.random()]]
        
        try:
            device = next(self.model.parameters()).device
            results = []
            
            for qa_pair in question_answer_pairs:
                # 准备输入
                if isinstance(qa_pair, dict):
                    text = self.config.format_str.format(**qa_pair)
                else:
                    text = str(qa_pair)
                
                # 获取模型输出
                inputs = self.tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    score = logits.mean().item()
                
                results.append([score])
            
            return results
        except Exception as e:
            print(f"奖励计算出错: {e}")
            return [[random.random()]] * len(question_answer_pairs)

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
qwen_prm = QwenPRM()  # 使用Qwen PRM模型

# 5. 创建CoT环境
cot_env = CoTEnv(
    config=config,
    math_problems=math_problems,
    llm_gen_fns=[deepseek_lm],
    rm_call=qwen_prm,
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

# 定义奖励函数
def reward_fn(x, verbose=False, legal_action=None):
    # 添加model_names参数，使用search_tree中定义的model_names
    return qwen_prm(x, model_names=["deepseek"])

results = search_tree.beam_search(
    simulate_env=cot_env,
    beam_size=2,
    max_step=3,
    reward_model_fn=reward_fn
)

# 7. 输出结果
for result in results:
    print(f"Answer: {result['text']}")
    print(f"Value: {result['value']:.2f}")
    print("="*50)