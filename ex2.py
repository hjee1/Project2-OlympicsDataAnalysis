import pandas as pd
import numpy as np
import scipy.stats as stats

# 카이제곱으로 데이터 분석해보기

data = pd.read_csv('team3_recentTenOlympic.csv')
print(data.head(4))

# 일단 필요없는 데이터 지우기 (drop 사용)
df = data.drop(['Sex', 'Name', 'Age', 'Height', 'Weight', 'Sport', 'Event', 'Season', 'Year'], axis=1)
print(df.head())  # type 은 dataframe

print()
print("______"*10)
print()

# 데이터 정리하기:
"""
에러로 많이 고생했던 코딩 흔적 ㅠㅠ
#df['Country_bi'] = df.loc[:,['NOC','Host']].apply(lambda n,h: 1 if n==h else )
#df['Country_bi'] = df[df.apply(lambda x: 1 if x['NOC']==x['Host'] else 0)]

#print(df.loc[:,['NOC','Host']].head())
"""
df['Country_bi'] = np.where(df["NOC"] == df["Host"], 1, 0)

print(df.head())

# 귀무가설 : 개최국인것과 메달은 관계가 없다.
# 대립가설 : 개최국인것과 메달은 관계가 있다.

print()
print("______"*10)
print()

# 빈도표
ctab = pd.crosstab(index=df['Country_bi'], columns=df['Medal_bi'])
print(ctab)

print()

chi2, p, ddof, expected = stats.chi2_contingency(ctab)
print('chi2:', chi2)  # 55.63413944379663
print('p:', p)  # 8.729439456355027e-14
print('ddof:', ddof)  # 1 : (2-1) * (2-1)

# 해석: p-value 8.729439456355027e-14 < 0.05 유의미한 수준에서 귀무가설을 기각, 대립가설을 채택 
# 결과: -  대립: 개최국인것과 메달은 관계가 있다 (독립적인 관계가 아니다)

# 기대효과: (얻을 수 있는 결론)
# 개최국의 이점을 생각하고 (안정감/시차), 
# 선수들의 컨디션을 최우선으로 생각하고 개최국으로 이동을 빠르게 하여 선수들이 현지국에서 적응을 빨리 할 수 있도록 도와준다. 
