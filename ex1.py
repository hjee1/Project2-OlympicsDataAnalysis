# 카이제곱으로 데이터 분석해보기

import pandas as pd
import numpy as np
import scipy.stats as stats

# 1 if c == "CHN" else 0
def country(d):
    if d['NOC'] == "CHN" and d['Games'] == "2002 Summer":
        return 1
    else:
        return 0
    
    

data = pd.read_csv('athlete_events.csv')
print(data.head(4))

# 일단 필요없는 데이터 지우기 (drop 사용)
df = data.drop(['Sex', 'Name', 'Age', 'Height', 'Weight', 'Sport', 'Event', 'Season', 'Year'], axis=1)
print(df.head(4))  # type 은 dataframe

print()
print("______"*10)
print()

# 메달을 딴 사람은 Medal_bi에 1을 주고
# 메달을 못딴 사람은 Medal_bi에 0을 준다

# print(df.info())
# df['Medal_bi'] = df['Medal'].apply(lambda m: 0 if m!=m else 1) # 이것도 가능
df['Medal_bi'] = df['Medal'].apply(lambda m: 0 if pd.isna(m) else 1)  # 이것도 가능

print(df.head(4))

print()
print("______"*10)
print()

# 주최국이 중국이라고 가정했을때: 중국인이면 1, 아니면 0

df['Country_bi'] = df['NOC'].apply(lambda c: 1 if c == "KOR" else 0)
print(df.head(4))

# 귀무가설 : 개최국이 항상 중국이라고 가정했을때, 개최국인것과 메달은 관계가 없다.
# 대립가설 : 개최국이 항상 중국이라고 가정했을때, 개최국인것과 메달은 관계가 있다.

nome = df[['Medal_bi','Country_bi']]
print(nome.head())

# 개최국 구분
no1 = nome[nome['Country_bi']==1] # 개최국
no2 = nome[nome['Country_bi']==0] # 비개최국
print(no1)
print(no2)

# 개최국 / 비개최국 별 메달 
me1 = no1['Medal_bi'] # 개최국이 딴 메달
me2 = no2['Medal_bi'] # 비개최국이 딴 메달
print(me1)
print(me2)
result = stats.ttest_ind(me1, me2) # equal_var = True : 등분산성을 만족한다는 의미
print(result) # pvalue=9.885881943229615e-21 < 0.05 이므로 귀무가설 기각, 대립 - 채택



