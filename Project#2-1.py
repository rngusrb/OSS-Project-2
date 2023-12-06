import pandas as pd
import pandas_datareader.data as web

def top10(data):
    for i in range(len(data)):
        hits = data[i].sort_values(by='H', ascending=False)[['batter_name', 'H']].head(10)
        avg = data[i].sort_values(by='avg', ascending=False)[['batter_name', 'avg']].head(10)
        homerun = data[i].sort_values(by='HR', ascending=False)[['batter_name', 'HR']].head(10)
        obp = data[i].sort_values(by='OBP', ascending=False)[['batter_name', 'OBP']].head(10)
        print(i+2015,"ranking\n\n",hits, "\n\n", avg, "\n\n", homerun, "\n\n", obp,'\n\n')


def bestwar():
    pos = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    for i in pos:
        best_war = d_2018[d_2018['cp'] == i].sort_values(by='war', ascending=False)[['batter_name', 'war']].head(1)
        print(i,' best player\n',best_war,'\n')

def salary_corr():
    corr_salary = frame[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']].corrwith(frame.salary).sort_values(ascending=False)
    print(corr_salary,"\n\n","highest correlation with salary is: ",corr_salary.head(1))


data=pd.read_csv('2019_kbo_for_kaggle_v2.csv')
frame=pd.DataFrame(data)
d_2015=frame[frame['year']==2015]
d_2016=frame[frame['year']==2016]
d_2017=frame[frame['year']==2017]
d_2018=frame[frame['year']==2018]
full_data=[d_2015,d_2016,d_2017,d_2018]

print('Q1: ')
top10(full_data)
print('Q2: 2018 best')
bestwar()
print('Q3: ')
salary_corr()