"""
matplotlib 绘图练习
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %matplotlib inline

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = 'SimHei'

def static_paint():
        """
        绘制静态图
        """
        
        df = pd.DataFrame({
                'city':['北京','北京','上海','上海','杭州','深圳','苏州','苏州','南京','广州'],
                'salary':[22.5,40,14,30,25,41.5,12.5,20,16,18]
        })

        fig,ax = plt.subplots(figsize=(9,6),sharey=True)
        # sns.barplot(x='city',y='salary',data=df,ci=95,ax=ax)
        # title 参数设置，让标题更好看
        ax.set_title('各个城市薪资水平对比', backgroundcolor='#3c7f99',fontsize=30, weight='bold',color='white')

        # 由于刻度的设置是tick的属性，所以用ax.tick_param()进行设置,用参数labelsize指定刻度标签的大小，将length参数设置刻度线长短。
        #  字体为16px大小，刻度线长度为0
        ax.tick_params(labelsize=16,length=0)
        plt.box(False)  #去掉四边的边框

        # 对各个城市的薪水均值并从小到大排序，获取城市排序列表city_order
        s = df.groupby(by=['city'])['salary'].agg('mean').sort_values()
        print(s)
        city_order = df.groupby(by=['city'])['salary'].agg('mean').sort_values().index.tolist()
        #排序后的绘图如下
        # Seaborn中的order和palette分别设置排列顺序和颜色。
        sns.barplot(x='city',y='salary',data=df,ax=ax,order=city_order,palette='RdBu_r')

        # 在y轴上添加网格线便于观察每个柱子的数值大小,因为是在y轴上，网格线为grid，所以用ax.yaxis.grid()进行设置
        ax.yaxis.grid(linewidth=0.5,color='black') # 设置y 轴网格线
        ax.set_axisbelow(True)         # 将网格线置于底层
        # 由于x轴和y轴含义比较清晰，所以可以将横纵坐标的标签去掉，同时，为了更直观，可以将y轴的刻度标签由20，15...换成20k,15k...
        ax.set_xlabel('')
        ax.set_ylabel('')
        # 将0处设为空字符串，其他地方加上k
        ax.set_yticklabels(['','5k','10k','15k','20k','35k','30k','35k'])
        # x轴标签更改
        plt.xticks(rotation='45', color='red',fontsize=10)
        plt.subplots_adjust(bottom=0.2)  #因为竖着字太长，生成图片中的x轴标签会被截取。因此设置距离底部0.2
        # plt.show()
        

if __name__ == "__main__":
    static_paint()