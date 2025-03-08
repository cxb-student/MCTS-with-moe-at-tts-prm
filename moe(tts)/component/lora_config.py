import torch
from peft import LoraConfig, TaskType, get_peft_model

def get_lora_config(r=8, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.CAUSAL_LM):
    """
    创建LoRA配置
    
    参数:
        r: LoRA的秩，较小的值会减少参数量但可能影响性能
        lora_alpha: LoRA的缩放参数
        lora_dropout: LoRA层的dropout率
        task_type: 任务类型，默认为因果语言模型
        
    返回:
        LoraConfig对象
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=task_type,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        modules_to_save=["lm_head", "embed_tokens"]
    )

def apply_lora_to_model(model, lora_config=None):
    """
    将LoRA应用到模型
    
    参数:
        model: 要应用LoRA的模型
        lora_config: LoRA配置，如果为None则使用默认配置
        
    返回:
        应用了LoRA的模型
    """
    if lora_config is None:
        lora_config = get_lora_config()
    
    # 检查模型是否已经应用了LoRA
    if hasattr(model, "peft_config"):
        print("模型已经应用了LoRA，跳过")
        return model
    
    # 应用LoRA
    try:
        peft_model = get_peft_model(model, lora_config)
        print(f"成功应用LoRA，参数数量: {peft_model.num_parameters()}")
        return peft_model
    except Exception as e:
        print(f"应用LoRA失败: {e}")
        return model