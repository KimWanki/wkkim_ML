import pandas as pd

data = pd.read_csv('/Users/kimwanki/Downloads/3405_6663_bundle_archive/movies_metadata.csv', index_col="title",
                   low_memory=False)
# index(세로)가 index_col에 설정한 column 값으로 변환됨.

print(data)
print(data.columns)

data = data[['genres', 'popularity', 'vote_average', 'vote_count']]
print(data)

df = pd.DataFrame({'Weight': [62, 67, 55, 74],
                   'Height': [165, 177, 160, 180],
                   }, index=['ID_1', 'ID_2', 'ID_3', 'ID_4'])

# calculate BMI
df.index.name = 'User'
print(df)
bmi = df['Weight'] / (df['Height'] / 100) ** 2
df['BMI'] = bmi
print(df)

# df.to_csv('/Users/kimwanki/Documents/GitHub/Wanki_ML/bmi.csv')
# 텍스트 파일을 보는 윈도우 명령어 !type
# !type /Users/kimwanki/Documents/GitHub/Wanki_ML/bmi.csv

index_list = ['P1001', 'P1002', 'P1003', 'P1004']
# TODO: DataFrame 데이터를 파일로 저장할 때 옵션을 지정하는 예제
df_pr = pd.DataFrame({'판매가격': [2000, 3000, 5000, 10000],
                      '판매량': [32, 53, 40, 25]
                      }, index=index_list)

df_pr.index.name = '제품번호'
print(df_pr)

# DataFrame 데이터를 텍스트 파일로 저장하기. 옵션으로 데이터 필드 : 공백으로 구분
file_name = '/Users/kimwanki/Documents/GitHub/Wanki_ML/save_DataFrame_cp949.txt'
df_pr.to_csv(file_name, sep=" ", encoding="cp949")

# !type '/Users/kimwanki/Documents/GitHub/Wanki_ML/save_DataFrame_cp949.txt'
