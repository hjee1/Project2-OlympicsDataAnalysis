# GDP와 메달 수 의 관계?


def prettyLine():
    print('_______' * 14)
    print()


import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('OlympicDataset.csv').drop(['Unnamed: 0'], axis=1)
print(df.head(3))

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________

print(df.columns)  # ['country', 'medals', 'Population', 'GDP']
print(np.corrcoef(df.medals, df.GDP))  # 0.44026533
# 상관계수 판단
print(df.corr())

"""
              medals  Population       GDP
medals      1.000000    0.205056  0.440265
Population  0.205056    1.000000 -0.090703
GDP         0.440265   -0.090703  1.000000
"""

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________

# boxplot으로 이상치 관찰:
"""
fig = plt.figure()
box1 = fig.add_subplot(1, 2, 1)
box2 = fig.add_subplot(1, 2, 2)
box1.boxplot(df.medals)
box2.boxplot(df.GDP)
plt.show()
"""

# 데이터의 퍼짐 정도를 시각화 (시각화의 중요성)
# plt.scatter(df.medals, df.GDP) # 산포도 표시
# plt.xlabel('medals')
# plt.ylabel('GDP')
# plt.show()

# 데이터에 있는 outlier 제거 고민...
# data = data.query('GDP <= 100000')
# data = data.query('medals <= 5000')

# 단순 선형 회귀 분석
model = smf.ols('medals ~ GDP + Population', data=df)
result = model.fit()
print(result.summary())
# R-squared:                       0.194
# Prob (F-statistic):           6.07e-07    --------> 유의한 모델
# slope = 0.0135    :     bias(intercept) = 25.6867
print('결정계수(설명력): ', result.rsquared) 
# 결정계수(설명력):  0.19383355740739927 -----> 설명력이 적다.

# 시각화
# 실제 값으로 산포도 표시
plt.scatter(df.GDP, df.medals)
# 회귀식을 화면에 표시
plt.plot(df.GDP, 0.0135 * df.GDP + 25.6867, 'r')  # Wx + B -> 기울기 * x + intercept / 예측 값으로 산포도 표시
plt.xlabel('GDP')
plt.ylabel('medals')
plt.show()

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________

# 예측 준비 완료:

"""
df.GDP = float(input('GDP를 입력하세요: '))
pred = result.predict(pd.DataFrame({'GDP':df.GDP}))
print('예상 총 메달 개수는:', int(pred[0]), ' 입니다.')
print(df.GDP)
"""

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________

import seaborn as sns

print('선형회귀분석모형의 적절성 확인: 정규성, 독립성, 선형성, 등분산성, 다중공선성 --------')

# 잔차항 (실제값 - 예측값) 구하기 (difference)
fitted = result.predict(df)  #  예측값
residual = df['medals'] - fitted  #  실제값 - 예측값
print('선형성 : 예측값과 잔차가 비슷한 패턴을 가짐')
sns.regplot(fitted, residual, line_kws={'color':'red'}, lowess=True)  # regplot(예측값, 잔차값)
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='grey')
plt.show()  # 완벽한 직선이 아니라서... 선형성을 완전하게 만족하지는 못함

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________

print('정규성 : 간차가 정규분포를 따라야함. Q-Q plot 사용')
import scipy.stats as stats
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x, y)
plt.plot([-3, 3], [-3, 3], '--', color='grey')
plt.show()  # 정규성을 완전하게 만족하지는 못함
print('shapiro test: ', stats.shapiro(residual)) 
# pvalue=2.010499007105984e-16 < 0.05 정규성을 만족 못함

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________

print('독립성 : 잔차가 독립적, 자기상관(인접 관측치와 오차가 상관되어있음) 이 없어야 함')
print('더빈왓슨 값으로 확인: Durbin-Watson:                   0.688')
# 더빈왓슨 값으로 확인: Durbin-Watson:                   0.688
# (0에 가까우면 양의 상관, 4에 가까우면 음의 상관. 2에 가까우면 자기상관이 없다)
# 그러므로 양의 상관관계, 독립성이 부족함

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________

print('등분산성 : 잔차의 분산이 일정')
sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess=True, line_kws={'color':'red'})
plt.show()  # 등분산성 만족 못합. 이상치 확인. 정규성, 선형성 확인

# ___________________________________________________________________________________
prettyLine()
# ___________________________________________________________________________________


print('다중 공선성 : 독립변수들 간에 강한 상관관계가 있는 경우')
# VIF(분산 인플레 요인) 값이 10을 넘으면 다중 공선성 발생
from statsmodels.stats.outliers_influence import variance_inflation_factor
print(model.exog_names)
print(variance_inflation_factor(model.exog, 1)) # 1.008295250855819 < 10
#여기 나온 수치가 VIF라는 값인데 해당 값이 10 이상일 경우 다중공선성이 발생한다.
