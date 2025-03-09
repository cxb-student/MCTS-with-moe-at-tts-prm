# MCTS_with_moe-prm

理论来源：

           1.prm的效果和问题的难度方向高度挂钩
           合理的分配比重能让结果更好
           2.moe-prm无法反向传播，所以肯定要用到强化学习的算法
           3.将gate视为actor，将整体saerch视为critic，按照deepseek的样式设计奖励规则

3.8发现问题：

           用prm的时候不论用什么都会过度思考，虽然答案可能正确，所以我认为要在moe上加上一个空白的情况，即原输出，来保证合理性
           已经编写好了相关代码（blank），但我感觉不会降低任何计算资源，只是形式好看，所以没跑过

实验流程（含计划）
 
           1.完善mcts方法
           2.构建moe架构
           3.grpo训练gate
           4.不同gate间实验
           5.基线测量


3.7-3.8 完成search方法的编写，借鉴于optimal tts的工作


![image](https://github.com/cxb-student/MCTS-with-moe-at-tts-prm/blob/main/random_test.png)

![image](https://github.com/cxb-student/MCTS-with-moe-at-tts-prm/blob/main/single.png)

           改进包含：去除无效代码
           测试：    用随机奖励替代prm运行以及加入一个prm，分别测试先行跑通
           
3.8     完成moe架构编写以及整体架构的集成
   
          包含：gate网络（注意力）
          Lora框架引入
          

3.8     阅读grpo文献，找到通用代码框架

3.9     github上找到grpo的现成代码

            减去了分布式训练的部分，并将math-verify的verify方法取缔
            加入lora框架，减轻内存消耗
            先测了一下基线
            数据集就是最简单的gsm8k
            因为会出bug，版本不兼容
            我将他分开写成三个文件，一个是单独的sft
            一个是只有一个prm的search来sft
            一个是moe架构的prm的search方法（未完成）
            应该先冷启动训练gate，之后再整体sft

![image](https://github.com/cxb-student/MCTS-with-moe-at-tts-prm/blob/main/train.png)
