# MCTS_with_moe-prm
（//MCTS_with_grpo 可能效果也不错//）

理论来源：

           1.prm的效果和问题的难度方向高度挂钩
           合理的分配比重能让结果更好
           2.moe-prm无法反向传播，所以肯定要用到强化学习的算法
           3.将gate视为actor，将整体saerch视为critic，按照deepseek的样式设计奖励规则

实验流程（含计划）
 
           1.完善mcts方法
           2.构建moe架构
           3.grpo框架引入
           4.多组实验分别验证（gate的效果，mcts的效果）
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

3.8     github上找到grpo的现成代码，借鉴于simple_grpo

            减去了分布式训练的部分，并将math-verify的verify方法取缔
            （版本不兼容咋也找不到文档）
            
            加入lora框架，减轻内存消耗
            先测了一下基线
            数据集就是最简单的gsm8k

            我将他分开写成三个文件
            
            一个是单独的sft
            一个是只有一个prm的search来sft（预计9号完成）
            一个是moe架构的prm的search方法（预计9号完成）


![image](https://github.com/cxb-student/MCTS-with-moe-at-tts-prm/blob/main/train.png)

3.9     单prm训练的想法，结合mcts和grpo的优点

         1.首先传统的反向传播肯定是不能够直接使用
         
         可以分别创建loss
         或者是顺着tragedy向前追溯，但这样只能够训练到llm而prm够不到
         
         2.关于奖励的方法以及整体的框架
         
         我有一个想法是利用grpo的最新成果
         就是我在search的最后一步不进行最优选择
         而是保留beam_size个候选答案，形成一组
         然后根据规则进行奖励（格式奖励，答案奖励和优势奖励）

3.9晚上  完成了单prm和moe——prm代码的编写，代码借鉴于 simple_grpo

         具体实现方法：
![image](https://github.com/cxb-student/MCTS-with-moe-at-tts-prm/blob/main/show_single.png)

         其实moe的区别跟这个不大，只是再prm整体的loss中还要归责
         然后通过规则判断出gate要选哪一个prm
         这样来形成标签来训练

         此外，联合的训练效果可能不稳定
         到时候可以删除一部分进行分块训练。

3.10凌晨

         lora框架已加入
         想调试发现跑不动7b的根本

