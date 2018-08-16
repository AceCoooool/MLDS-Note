# ch3. Generalization

这章主要从两个方面来思考Deep Learning的**泛化能力**：

1. 从实验的角度来看看模型复杂度与泛化能力之间的关系

   - Deep Learning模型的复杂度（capacity）很大

   - Deep Learning模型的复杂度（capacity）与泛化能力之间的关系和"常规"机器学习模型的复杂度与泛化能力之间的关系有些不同（模型越复杂往往并不会造成overfitting，且能够提高testing accuracy）

2. 可以通过一些指标来衡量一个模型的泛化能力如何

   - Sensitivity：在"某一点"（某一个testing data）上面的Sensitivity越小，往往暗示着该点的结果会比较好，而如果在某点上面的Sensitivity越大，说明该模型在该点上面的泛化能力并不好（可能该点在训练集里没有相似的对象）
   - Sharpness（这个指标暂时存疑）：当training阶段的解为比较平坦的local minima时候（flatness），往往其泛化能力比较好；而那些比较尖锐的local minima（Sharpness），其泛化能力一般不太好（可能由训练集与测试集分布的不一致导致的）--- 与此相关的一个概念就是Small Batch Size越倾向于走到更flatness的local minima