# /*
#  * @Author: wangchong 
#  * @python: 3.6
#  * @Date: 2022-10-24 22:19:59 
#  * @Last Modified by:   wangchong 
#  * @Last Modified time: 2022-10-24 22:19:59 
#  */

## cuDF是pandas的gpu版本,可以在gpu上运行


import pandas as pd

# 加载数据
df = pd.read_csv('D:\VS_Space\Pandas Learning Note\data\gapminder.tsv', 
                sep='\t',header=0, names=None)

# 查看数据的基本信息
print(df.head())
print(type(df))
print(df.shape)
print(df.columns)
print(type(df.columns))
print(df.dtypes)
print(df.info())

print(df.columns)
print(df.index)


# 获取列子集
country_df = df['country']
print(country_df.head())
print(country_df.tail())
subset = df[['country', 'continent', 'year']]
print(subset.head())
print(subset.tail())
df[[1]]

# 获取行子集
print(df.loc[0])
print(df.loc[99])
print(df.loc[-1])
print(df.loc[df.shape[0] - 1])
print(df.tail(n=1))

subset_loc = df.loc[0]
subset_head = df.head(n=1)
print(type(subset_loc))
print(type(subset_head))

print(df.loc[[0, 99, 999]])

print(df.iloc[1])
print(df.iloc[99])
print(df.iloc[-1])
print(df.iloc[[0, 99, 999]])

# 混合
## 获取列子集
subset = df.loc[[0, 1, 2, 3], ['year', 'pop']]
subset.index
subset.index = ['one', 'two', 'three', 'four']
subset.index
subset.loc['one']
subset.iloc['one']
subset.iloc[0]

subset = df.iloc[:, [2, 4, -1]]

## 通过范围获取列子集
subset = df.iloc[:, list(range(5))]
subset = df.iloc[:, list(range(3, 6))]
subset = df.iloc[:, list(range(10))]

## 使用切片语法获取子集
subset = df.iloc[:, :3]
subset = df.iloc[:, 3:6]
subset = df.iloc[:, 0:6:2]

df.iloc[:, 0:6:] #默认步长为1
df.iloc[:, 0::2] #省略列的总数，在不知道列数时使用
df.iloc[:, :6:2] #从第0列开始
df.iloc[:, ::2]  #只取偶数列
df.iloc[:, ::]   #取全部列

## 获取行和列的子集
print(df.loc[42, 'country'])
print(df.iloc[42, 0])
print(df.loc[42, 0])

## 获取多行和多列
print(df.iloc[[0, 99, 999], [0, 3, 5]])
print(df.loc[[0, 99, 999], ['country', 'lifeExp', 'gdpPercap']])
print(df.loc[10:13, ['country', 'lifeExp', 'gdpPercap']])

## 获取符合条件的行和列
df.loc[df['year'] == 1952].head(5)
df.loc[(df['year'] == 1952) & (df['country'] == 'Albania')]
df.loc[df['year'].isin([1952, 1957])].head(5)
df.loc[(df['year'] == 1952) | (df['country'] == 'Albania')]


# 分组和聚合计算
print(df.head(n=10))
## 分组方式
print(df.groupby('year')['lifeExp'].mean())
grouped_year_df = df.groupby('year')
print(type(grouped_year_df))
print(grouped_year_df)

grouped_year_df_lifeExp = grouped_year_df['lifeExp']
print(type(grouped_year_df_lifeExp))
print(grouped_year_df_lifeExp)
mean_lifeExp_by_year = grouped_year_df_lifeExp.mean()
print(mean_lifeExp_by_year)

multi_group_var = df.groupby(['year', 'continent'])\
                            [['lifeExp', 'gdpPercap']].mean()
print(multi_group_var)

flat = multi_group_var.reset_index()
print(flat.head(15))

## 分组频率计数
print(df.groupby('continent')['country'].nunique())
df['country'].value_counts()
df['continent'].value_counts()

# 基本绘图
import matplotlib.pyplot as plt

global_yearly_life_expectancy = df.groupby('year')['lifeExp'].mean()
print(global_yearly_life_expectancy)
global_yearly_life_expectancy.plot()
plt.show()

