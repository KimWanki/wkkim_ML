# !conda activate DS
import pandas as pd

# df = pd.DataFrame(data [, index = index_data, columns= columns_data])
# 세로축 라벨 : index , 가로축 라벨 : columns

#           -> columns
#         -----------
#   |   ㅣ value1  value2
#   V   ㅣ value3  value4
# index

df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(df)

index_list = pd.date_range('2020.07.28', periods=3)
columns_list = ['A', 'B', 'C']

df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=index_list, columns=columns_list)
# print(df2)

table_data = {'연도': [2015, 2016, 2017, 2017, 2018],
              '지사': ['한국', '한국', '미국', '한국', '미국'],
              '고객 수': [200, 250, 450, 300, 500]}

# print(table_data)
#
# print(list(table_data.keys()))
# print(list(table_data.items()))

df = pd.DataFrame(table_data)
# print(df)
# print(df.values)

print(df.index)
print(df.columns)
print(df.values)

s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([10, 20, 30, 40, 50])
# Series()로 생성한 데이터 사칙연산
# 덧셈
s3 = s1 + s2
print(s3)
# 뺄셈
s4 = s2 - s1
print(s4)
# 곱셈
mul = s1 * s2
print(mul)

table_data1 = {'A': [1, 2, 3, 4, 5],
               'B': [10, 20, 30, 40, 50],
               'C': [100, 200, 300, 400, 500]}
df1 = pd.DataFrame(table_data1)
print(df1)

table_data2 = {'A': [6, 7, 8],
               'B': [60, 70, 80],
               'C': [600, 700, 800]}
df2 = pd.DataFrame(table_data2)
print(df2)

result = df1 + df2
print(result)

# numpy와 달리 사이즈가 달라도 사칙연산이 가능하지만, 연산이 되지 않는 경우, NaN으로 표기된다.


# 강수량에 관한 데이터
table_data3 = {'봄': [256.5, 264.3, 215.9, 223.2, 312.8],
               '여름': [770.6, 567.5, 599.8, 387.1, 446.2],
               '가을': [363.5, 231.2, 293.1, 247.7, 381.6],
               '겨울': [139.3, 59.9, 76.9, 109.1, 108.1]}

columns_list = list(table_data3.keys())
index_list = ['2012', '2013', '2014', '2015', '2016']
df3 = pd.DataFrame(table_data3, columns=columns_list, index=index_list)

# 계절별 강수량
print(df3)
print(df3.mean())
print(df3.std())

# axis = 0인 경우, values를 열별로 연산을 수행. 1이면 행별로 연산을 수행
# 설정하지 않는 경우 기본값으로 0이 설정.

# 연도별 강수량
print(df3.mean(axis=1))
print(df3.std(axis=1))

print(df3.describe())

