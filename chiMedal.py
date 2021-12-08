import pandas as pd
import numpy as np
import scipy.stats as stats

data = pd.read_csv('../testdata/athlete_events.csv')
print(data.head(3))
ath_df = data.drop(['Sex','Name','Age','Height','Weight','Games','Sport','Event'], axis =1)
print(ath_df.head(3))

print(ath_df['Medal'])
ath_df['Medal'] = ath_df['Medal'].apply(lambda m:1 if m =='Gold' or m =='Silver' or m =='Bronze'
                                        else 0) # 금은동메달은 1, 이를 제외한 4위 이하는 0
print(ath_df['Medal'].head(50))

ath_df['NOC'] = ath_df['NOC'].apply(lambda m:1 if m =='CHN' else 0) # 중국을 개최국으로 가정함
print(ath_df.head(50))

# 귀무 : 개최국(중국)은 좋은 성과를 낼 수 있다
# 대립 : 개최국은 좋은 성과를 낼 수 없다

nome = ath_df[['Medal','NOC']]
print(nome.head())

# 개최국 구분
no1 = nome[nome['NOC']==1] # 개최국
no2 = nome[nome['NOC']==0] # 비개최국
print(no1)
print(no2)

# 개최국 / 비개최국 별 메달 
me1 = no1['Medal'] # 개최국이 딴 메달
me2 = no2['Medal'] # 비개최국이 딴 메달
print(me1)
print(me2)
result = stats.ttest_ind(me1, me2) # equal_var = True : 등분산성을 만족한다는 의미
print(result) # pvalue=9.885881943229615e-21 < 0.05 이므로 귀무가설 채택
# 따라서 개최국(중국)은 좋은 성과를 낼 수 있다