反向传播的输出误差：
```math
\delta_k = (y - y_k)f^\prime(h_k)
```
- `$y$` 表示目标值，`$ y_k $`表示训练后输出层的输出值
- `$h_k$` 表示输出层的输入值，即 `$h_k = \sum x_iw_i$`
- `$f^\prime(h_k)$` 表示求导

根据输出误差可以求出隐藏层的误差：
```math
\delta_j = \sum\delta_kw_{jk}f^\prime(h_j)
```
- `$\delta_k$`表示输出误差
- `$w_{jk}$`表示隐藏层到输出层之间的权重
- `$f^\prime(h_j)$`表示隐藏层的输入值的导数值

隐藏层的权重更新值：
```math
\Delta w_{ij} = \eta*\delta^k_j * x_i
```
 - 权重更新值等于输出误差乘以隐藏层的输出值
 - `$\eta$`表示学习率,是一个常数
 - `$\delta^k_j $`表示输出层的误差
 - `$x_i$`表示隐藏层的输出值
 
 


![image](https://mmbiz.qpic.cn/mmbiz_png/rw1wCRwDbgYrcRQQaT2WrAEefWz8F0b8ibcfEEgzWwgOORYC8Uw0hD8xn9PwHjDNlwIPqH6IosMG3DP2kzfr4yg/0?wx_fmt=png)


---
##### 反向传播实例
```py
import numpy as np
from data_prep import features, targets, features_test, targets_test #data import

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    """
    Calculate sigmoid
    """
    return sigmoid(x) * (1 - sigmoid(x))

# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x,weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        o_input = np.dot(hidden_output,weights_hidden_output)
        output = sigmoid(o_input)

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        # TODO: Calculate error term for the output unit
        output_error_term = error * sigmoid_prime(o_input)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term , weights_hidden_output)
        
        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * sigmoid_prime(hidden_input)
        
        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:,None] 

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))



```





