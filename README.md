# MCTS_with_moe-prm
理论来源：

           1.prm的效果和问题的难度方向高度挂钩
           合理的分配比重能让结果更好
           2.moe-prm无法反向传播，所以肯定要用到强化学习的算法
           3.将gate视为actor，将整体saerch视为critic，按照deepseek的样式设计奖励规则
           
实验流程（含计划）
 
           1.完善mcts方法
           2.构建moe架构
           3.grpo训练gate
           4.不同gate间实验
           5.基线测量


3.7-3.8 完成search方法的编写，借鉴于optimal tts的工作


![image](https://github.com/cxb-student/MCTS-with-moe-at-tts-prm/blob/main/image.png)

           改进包含：去除无效代码
           测试：    用随机奖励替代prm运行，先行跑通
           
3.8     完成moe架构编写以及整体架构的集成
   
          包含：gate网络（注意力）
          Lora框架引入
          

3.8     阅读grpo文献，找到通用代码框架
