import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

#多项式拟合正弦曲线
a = np.arange(0, 2*np.pi, 0.1)
y_sinx = np.sin(a)
x = np.arange(0, 6, 0.6).reshape(10, -1)
np.random.seed(21)
epison = np.random.normal(0, 1, size=x.shape)
y_true = np.sin(x) + epison

plt.plot(a, y_sinx, label="y_sinx")
plt.scatter(x, y_true, label="y_true", color="r")
plt.legend()
plt.show()


#二次多项式
pipe = Pipeline([('feature', PolynomialFeatures(degree=2)),
                 ('lr', LinearRegression(fit_intercept=False))])
pipe.fit(x, y_true)
coef_2 = pipe['lr'].coef_
y_2 = pipe.predict(x)
#三次多项式
pipe = Pipeline([('feature', PolynomialFeatures(degree=3)),
                 ('lr', LinearRegression(fit_intercept=False))])
pipe.fit(x, y_true)
coef_3 = pipe['lr'].coef_
y_3 = pipe.predict(x)
#五次多项式
pipe = Pipeline([('feature', PolynomialFeatures(degree=5)),
                 ('lr', LinearRegression(fit_intercept=False))])

pipe.fit(x, y_true)
coef_5 = pipe['lr'].coef_
y_5 = pipe.predict(x)
#九次多项式
pipe = Pipeline([('feature', PolynomialFeatures(degree=9)),
                 ('lr', LinearRegression(fit_intercept=False))])

pipe.fit(x, y_true)
coef_9 = pipe['lr'].coef_
y_9 = pipe.predict(x)

#绘制各多项式的图像
fig = plt.figure(figsize=(2, 2))
axes1 = fig.add_subplot(2, 2, 1)
axes2 = fig.add_subplot(2, 2, 2)
axes3 = fig.add_subplot(2, 2, 3)
axes4 = fig.add_subplot(2, 2, 4)

axes1.plot(a, y_sinx, label="y_sinx")
axes1.scatter(x, y_true, label="y_true", color="r")
axes1.plot(x, y_2, label="M=2", color="green")
axes1.legend()
axes2.plot(a, y_sinx, label="y_sinx")
axes2.scatter(x, y_true, label="y_true", color="r")
axes2.plot(x, y_3, label="M=3", color="green")
axes2.legend()
axes3.plot(a, y_sinx, label="y_sinx")
axes3.scatter(x, y_true, label="y_true", color="r")
axes3.plot(x, y_5, label="M=5", color="green")
axes3.legend()
axes4.plot(a, y_sinx, label="y_sinx")
axes4.scatter(x, y_true, label="y_true", color="r")
axes4.plot(x, y_9, label="M=9", color="green")
axes4.legend()
fig.tight_layout()
fig.show()


pipe = Pipeline([('feature', PolynomialFeatures(degree=9)),
                 ('ridge', Ridge(alpha=0.01, fit_intercept=False))])

pipe.fit(x, y_true)
l2_coef_9 = pipe['ridge'].coef_
y_9 = pipe.predict(x)

np.set_printoptions(suppress=True)
print(coef_9)

