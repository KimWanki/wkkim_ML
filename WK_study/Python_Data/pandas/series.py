import pandas as pd

s1 = pd.Series([10, 20, 30, 40, 50])
print(s1)

s2 = pd.Series(['a', 'b', 'c', 1, 2, 3])
print(s2)

print(s2.values)

import numpy as np

s3 = pd.Series([np.nan, 10, 30])

index_date = ['2018-10-07', '2018-10-08', '2018-10-09', '2018-10-10']
s4 = pd.Series([200, 195, np.nan, 205], index=index_date)

s5 = pd.Series({'국어': 100, '영어': 95, '수학': 90})
print(s5)

#   s = pd.Series(dict_data)
#   딕셔너리 이용시 data와 index 동시 입력 가능함.

date = pd.date_range(start='2019-01-01', end='2019-01-08')
date2 = pd.date_range(start='2020/01/03', periods=8)
print(date)
print(date2)
