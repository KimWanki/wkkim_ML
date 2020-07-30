import pandas as pd
import numpy as np

KTX_data = {'경부선 KTX': [39060, 39896, 42005, 43621, 41702, 41266, 32477],
            '호남선 KTX': [7315, 4872, 1782, 5893, 2278, 7891, 2789],
            '경전선 KTX': [7892, 4327, 5829, 3678, 4895, 2781, 3678],
            '전라선 KTX': [237, 1782, 1987, 2890, 2843, 5782, 3427],
            '동해선 KTX': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}
col_list = list(KTX_data.keys())
index_list = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']

df_KTX = pd.DataFrame(KTX_data, columns=col_list, index=index_list)
print(df_KTX)


# DataFrame 데이터 분석 시 너무 많은 데이터가 출력돼 분석할 때 오히려 불편할 수 있음.

# pandas에서는 DataFrame의 head와 tail만 반환 가능함.
# 형식 : DataFrame_data.head([n]
print(df_KTX.head())
print(df_KTX.tail())
print(df_KTX.loc['2011':'2014'])

df_A_B = pd.DataFrame({'판매월': ['1월', '2월', '3월', '4월'],
                       '제품A': [100, 150, 200, 130],
                       '제품B': [90, 110, 140, 170]})
df_C_D = pd.DataFrame({'판매월': ['1월', '2월', '3월', '4월'],
                       '제품C': [112, 141, 203, 134],
                       '제품D': [90, 110, 140, 170]})
print(df_A_B)
print(df_C_D)

# 데이터가 모두 값을 가지고 있을 경우, 통합하는 방법
# on 인자에는 통합하려는 기준이 되는 특정 열(key)의 라벨 이름(key_label)을 입력.
print(df_A_B.merge(df_C_D))

df_left = pd.DataFrame({'key': ['A', 'B', 'C'], 'left': [1, 2, 3]})
df_right = pd.DataFrame({'key': ['A', 'B', 'D'], 'right': [4, 5, 6]})

print(df_left.merge(df_right, how='outer'))
print(df_left.merge(df_right, how='inner'))
