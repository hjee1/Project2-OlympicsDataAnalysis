import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

# 데이터 읽기, 및 데이터 확인
summer_data = pd.read_csv("summer.csv")
winter_data = pd.read_csv("winter.csv")
dic_data = pd.read_csv("dictionary.csv")

# 데이터 병합 코드
frame = [summer_data, winter_data]
data = pd.concat(frame)
#print(data.columns)
#print(data)

series = data.groupby(['Country']).Medal.count()
df = pd.DataFrame({'country':series.index, 'medals':series.values})
#print(df)
#print(type(df))

print()
final_df = pd.merge(df, dic_data, left_on='country', right_on='Code').drop(['Code', 'Country'], axis=1)

# Nan행 제거
final_df = final_df.dropna(how='any')
final_df = final_df.sort_values(by=['medals'], ascending=False)
print(final_df)

# csv 파일로 저장
final_df.to_csv('OlympicDataset.csv', sep=',')








