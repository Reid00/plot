import numpy as np
import matplotlib.pyplot as plt

# 折线图
x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linar')
plt.plot(x, x ** 2, label='quadratic')
plt.plot(x, x ** 3, label='cubic')

plt.xlabel('x label content')
plt.ylabel('y label content')

plt.title('simple plot')
plt.legend()
plt.show()

# 散点图

x = np.arange(0, 5, 0.2)
# 红色破折号, 蓝色方块 ，绿色三角块
plt.plot(x, x, 'r--', x, x ** 2, 'bs', x, x ** 3, 'g^')

plt.show()

# 直方图
np.random.seed(19680801)
mu1, singma1 = 100, 15
mu2, singma2 = 80, 15
x1 = mu1 + singma1 * np.random.randn(10000)
x2 = mu2 + singma2 * np.random.randn(10000)
# the histogram of the data
# 50：将数据分成50组
# facecolor：颜色；alpha：透明度
# density：是密度而不是具体数值
n1, bins1, patches1 = plt.hist(x1, 50, density=True, facecolor='g', alpha=1)
n2, bins2, patches2 = plt.hist(x2, 50, density=True, facecolor='r', alpha=0.2)

# n：概率值；bins：具体数值；patches：直方图对象。
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')

plt.text(110, 0.25, r'$\mu=100,\ \sigma=15$')
plt.text(50, 0.25, r'$\mu=80,\ \sigma=15$')

# 设置x，y轴的具体范围
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

# 并列柱状图
size = 5
a = np.random.random(size)
b = np.random.random(size)
c = np.random.random(size)
x = np.arange(size)
# 有多少个类型，只需更改n即可
total_width, n = 0.8, 3
width = total_width / n
# 重新拟定x的坐标
x = x - (total_width - width) / 2
# 这里使用的是偏移
plt.bar(x, a, width=width, label='a')
plt.bar(x + width, b, width=width, label='b')
plt.bar(x + 2 * width, c, width=width, label='c')

plt.legend()
plt.show()

# 叠加柱状图
size = 5
a = np.random.random(size)
b = np.random.random(size)
c = np.random.random(size)

x = np.arange(size)

# 这里使用的是偏移
plt.bar(x, a, width=0.5, label='a', fc='r')
plt.bar(x, b, bottom=a, width=0.5, label='b', fc='g')
plt.bar(x, c, bottom=a + b, width=0.5, label='c', fc='b')

plt.ylim(0, 2.5)
plt.legend()
plt.grid(True)
plt.show()

# 普通饼图

labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
size = [15, 30, 45, 10]
# 设置分离的距离，0表示不分离
explode = (0, 0.1, 0, 0)
plt.pie(size, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
# Equal aspect ratio 保证画出的图是正圆形
plt.axis('equal')
plt.show()

# 嵌套饼图
# 设置每环的宽度
size = 0.3
vals = np.array([[60., 32.], [37., 40.], [29., 10.]])

# 通过get_cmap随机获取颜色
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3) * 4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

print(vals.sum(axis=1))
# [92. 77. 39.]

plt.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'))
print(vals.flatten())
# [60. 32. 37. 40. 29. 10.]

plt.pie(vals.flatten(), radius=1 - size, colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor='w'))

# equal 使得为正圆
plt.axis('equal')
plt.show()