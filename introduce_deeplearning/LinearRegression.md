- 传统编程：确定每一个步骤，得到一个结果
- 机器学习：先定义结果（只要你能想到的就有可能实现），然后程序进行自我学习，寻找步骤得到结果

机器学习的学习方式：
- 监督学习：带有标签，每一个步骤都有反馈
- 无监督学习：没有标签，没有反馈（即使最终也没有反馈）
- 强化学习：只有达到目的才有反馈，不断试错，与环境互动

深度学习是机器学习的一个子集，它几乎跑赢了其他所有模型（马尔可夫链，SVM，决策树）

---
简单线性回归-使用 BMI 来预测平均寿命:
```py
# TODO: Add import statements
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]


# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values,y_values)

#visualize results
plt.scatter(x_values,y_values)
plt.plot(x_values,bmi_life_model.predict(x_values))
plt.show()

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931 )
print(laos_life_exp)

```
---

**线性回归注意事项**：
- 线性回归会根据训练数据生成直线模型。如果训练数据包含非线性关系，你需要选择：调整数据（进行数据转换）、增加特征数量或改用其他模型。
- 如果数据集中存在不符合总体规律的异常值，最终结果将会存在不小偏差。