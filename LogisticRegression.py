import numpy as np
import matplotlib.pylab as plt
import math

m = 100  # train_num
n = 2  # class_num
sigma = 1.5
mu = 3

Y = np.random.randint(0, 2, (m, 1))
X_array = np.ones(shape=(m, (n + 1)))
X = np.mat(X_array)
# 生成数据的时候就要分开啊！！！！！！！调了一天bug竟然是这的问题！！！
X[:, 1] = sigma * np.random.randn(m, 1) + mu + 3 * Y
X[:, 2] = sigma * np.random.randn(m, 1) + mu + 3 * Y

theta_array = np.ones(shape=((n + 1), 1))
theta = np.mat(theta_array)
for i in range(0, n + 1):
    theta[i, 0] = 1.0


def h(X):
    return 1.0 / (1 + np.exp(-X))


J = []  # loss函数
oneY = np.mat(np.ones(shape=(m, 1)))

# #梯度下降法
# error = np.mat(np.ones(shape=(m, 1)))
# iterNum = 30000
# alpha = 0.05
# lamda = 0.01
# deacy = 0.1
# for i in range(0, iterNum):
#     predict = h(X * theta)
#     error = predict - Y
#     # lr = alpha / (1 + (i + 1) * deacy)
#     # theta = theta - lr * (1.0 / m) * X.T * error
#     theta = theta - alpha * (1.0 / m) * X.T * error
#     # theta = theta - alpha * (1.0 / m) * X.T * error + lamda*theta
#     loss = (-1.0 / m * (Y.T * np.log(h(X * theta)) +
#                         (oneY - Y).T * np.log((oneY - h(X * theta)))))
#     J.append(loss[0, 0])

# 牛顿法
iterNum = 10000
alpha = 0.001
A = np.mat(np.ones(shape=(m, m)))

for i in range(0, iterNum):
    predict = h(X * theta)
    error = predict - Y
    U = -X.T * error
    Aa = np.multiply(h(X * theta), (h(X * theta) - 1))
    A = np.diag(Aa.T.A1)
    # for j in range(0, m):
    #     A[j, j] = h(X[j] * theta) * \
    #               (h(X[j] * theta) - 1)
    H = X.T * np.mat(A) * X
    theta = theta - alpha*H.I * U
    p = h(X * theta)
    loss = (-1.0 / m * (Y.T * np.log(p)
                        + (oneY - Y).T * np.log((oneY - p))))
    # if loss[0, 0] < 0.2:
    #     print("real iterNum:", i)
    #     break
    J.append(loss[0, 0])

print(J)

plt.figure(1, dpi=100)
plt.figure()
plt.xlabel("X axis")
plt.ylabel("Y axis")
color = ['c' if y == 1 else 'r' for y in Y]
# list don't need a new array
plt.scatter(X[:, 1].tolist(), X[:, 2].tolist(), c=color, alpha=0.4)
x = np.linspace(-2, 8, 100)
y = 1.0 * (-theta[0, 0] - theta[1, 0] * x) / theta[2, 0]
plt.plot(x, y, color="b", label='classify')
plt.legend()
# 去掉右边框和上边框
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 显示所画的图
plt.show()

plt.figure(2, dpi=100)
x = range(0, iterNum)
y = J
plt.plot(x[8000:], y[8000:], color="k", label='loss')
plt.legend()
# 去掉右边框和上边框
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 显示所画的图
plt.show()
