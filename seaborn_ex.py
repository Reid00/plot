import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#pandas matplotlib https://mp.weixin.qq.com/s/Pg6iklcNmtUp0np1yeKIlg
# sns.set()

# 1. 单变量分布可视化(displot)
# 2. 双变量分布可视化(jointplot)
# 3. 数据集中成对双变量分布(pairplot)
# 4. 双变量-三变量散点图(relplot) relational plots
# 5. 双变量-三变量连线图(relplot)
# 6. 双变量-三变量简单拟合
# 7. 分类数据的特殊绘图(catplot -> categorical plots) 

# 单变量分布可视化
# distplot()对单变量分布画出直方图(可以取消)，并自动进行概率分布的拟合(也可以使用参数取消)。
# 直方图，密度图为了展示单变量的分布情况，即分布概率
# 直方图并拟合核密度估计图。
sns.set_style('darkgrid')
x=np.random.randn(200)
iris=sns.load_dataset('iris')
y=iris['sepal_length']
sns.distplot(x)
sns.distplot(y)
plt.savefig('distplot.png')
plt.show()
# kde=False 则可以只绘制直方图 hist=False 不显示直方图
sns.distplot(x,hist=False)
plt.savefig('distplot2.png')
sns.distplot(x,kde=False)
plt.show()


# 双变量分布
# 双变量分布通俗来说就是分析两个变量的联合概率分布和每一个变量的分布。
mean,cov=[0,1],[(1,0.5),(0.5,1)]
data=np.random.multivariate_normal(mean,cov,200)
data=pd.DataFrame(data,columns=['xdata','ydata'])
# print(data)
sns.jointplot(x='xdata',y='ydata',data=data)
plt.savefig('jointplot.png')
# 同样可以使用曲线来拟合分布密度
sns.jointplot(x='xdata', y='ydata',data=data, kind='kde')
plt.show()


# 数据集中成对双变量分析
# 对于数据集有多个变量的情况，如果每一对都要画出相关关系可能会比较麻烦，利用Seaborn可以很简单的画出数据集中每个变量之间的关系。
iris= sns.load_dataset('iris')
# 通过指定hue来对数据(columns name)进行分组(效果通过颜色体现)
# 通过指定vars=["sepal_width", "sepal_length"]显式展示指定变量名对应的数据
print(iris.head())
sns.pairplot(iris,hue='species',palette='husl')
plt.savefig('pairplot.png')
sns.pairplot(iris,hue='species',palette='husl', vars=['sepal_width','sepal_length'])
sns.pairplot(iris,palette='husl', vars=['sepal_width','sepal_length'])
plt.show()


# 双变量-三变量散点图
# 统计分析是了解数据集中的变量如何相互关联以及这些关系如何依赖于其他变量的过程，有时候在对数据集完全不了解的情况下，可以利用散点图和连线图对其进行可视化分析，这里主要用到的函数是relplot函数。
tips= sns.load_dataset('tips')
print(tips.head())
sns.relplot(x='total_bill',y='tip', data=tips)
plt.show()
# 除了画出双变量的散点图外，还可以利用颜色来增加一个维度将点分离开
sns.relplot(x='total_bill',y='tip',data=tips,hue='smoker')
plt.show()
# 为了强调数据之间的差异性，除了颜色也可以使用图形的不同来分类数据点（颜色和形状互相独立）, kind=line 表示线性图
sns.relplot(x="total_bill", y="tip", hue="smoker", style="smoker",data=tips)
plt.savefig('relplot.png')
plt.show()


# 线性模型可视化
# 主要用regplot()进行画图，这个函数绘制两个变量的散点图，x和y，然后拟合回归模型并绘制得到的回归直线和该回归一个95％置信区间。
sns.set_style('darkgrid')
sns.regplot(x='total_bill',y='tip',data=tips)
plt.show()
sns.regplot(x='size',y='tip',data=tips)
plt.show()


# 拟合不同类型的模型
# 线性模型对某些数据可能适应不够好，可以使用高阶模型拟合
anscombe= sns.load_dataset('anscombe')
sns.regplot(x='x',y='y',data=anscombe.query("dataset == 'II'"),ci=None)
plt.show()



# 分类散点图
# 可以使用两种方法来画出不同数据的分类情况，第一种是每个类别分布在对应的横轴坐标上，而第二种是为了展示出数据密度的分布从而将数据产生少量随即抖动进行可视化的方法。
# 微小抖动来展示出数据分布
sns.catplot(x='day',y='total_bill',data=tips)
plt.show()
# 利用jitter来控制抖动大小或者是否抖动
sns.catplot(x='day',y='total_bill',jitter=False,data=tips)
plt.show()
# 同时可以使用swarm方法来使得图形分布均匀
sns.catplot(x='day',y='total_bill',kind='swarm',data=tips)
plt.show()
# 值得注意的是，与上面的scatter相同，catplot函数可以使用hue来添加一维，但是暂不支持style
sns.catplot(x='day',y='total_bill',hue='sex',kind='swarm',data=tips)
plt.savefig('swarm.png')
plt.show()


# 分类分布图
# 随着数据的增加，分类数据的离散图更为复杂，这时候需要对每类数据进行分布统计。这里同样使用高级函数catplot()。
# 箱线图
# 显示了分布的三个四分位数值以及极值
sns.catplot(x='day',y='total_bill',kind='box',data=tips)
plt.show()
# 同样可以使用hue来增加维度
# 箱线图，展示数据的四分位数，检测数据是否有极端值。最两端是最大内限
sns.catplot(x='day',y='total_bill',hue='smoker',kind='box',data=tips)
plt.savefig('box.png')
plt.show()
# 小提琴图事实上是密度图和箱型图的结合
# 分别表示箱型图的含义和任意位置的概练密度
sns.catplot(x='day',y='total_bill',hue='time',kind='violin',data=tips)
plt.savefig('violin.png')
plt.show()
# 当hue参数只有两个级别时，也可以“拆分”小提琴，这样可以更有效地利用空间
sns.catplot(x='day',y='total_bill',hue='sex',kind='violin',split=True,data=tips)
plt.show()



# 分类估计图,变化趋势
# 如果我们更加关心类别之间的变化趋势，而不是每个类别内的分布情况，同样可以使用catplot来进行可视化。
# 条形图，利用bar来画出每个类别的平均值.阵条的高度反映数值变量的集中趋势
# 黑色表示估计区间,即置信区间
titanic= sns.load_dataset('titanic')
sns.catplot(x='sex',y='survived',hue='class',kind='bar',data=titanic)
plt.savefig('bar.png')
plt.show()
# 如果更加关心的是类别的数量而不是统计数据的话可以使用count
sns.catplot(x='deck',kind='count',data=titanic)
plt.savefig('count.png')
plt.show()
# 点图功能提供了一种可视化相同信息的替代方式
# 只画出估计值和区间，而不画出完整的条形图
sns.catplot(x='sex',y='survived',hue='class',kind='point',data=titanic)
plt.show()
# 更加复杂的设置，通过对每一个参数进行设置来调节图像
sns.catplot(x='class',y='survived',hue='sex',
palette={'male':'g','female':'m'},
markers=['^','o'],
linestyles=['-','--'],
kind='point',
data=titanic
)
plt.show()


# 矩阵图
# 矩阵图中最常用的就只有 2 个，分别是：heatmap 和 clustermap。
# 热力图在某些场景下非常实用，例如绘制出变量相关性系数热力图。
# 除此之外，clustermap 支持绘制层次聚类结构图。如下所示，我们先去掉原数据集中最后一个目标列，传入特征数据即可。当然，你需要对层次聚类有所了解，否则很难看明白图像多表述的含义。
sns.heatmap(np.random.rand(10, 10),annot=True)
plt.savefig('heatmap.png')
plt.show()
