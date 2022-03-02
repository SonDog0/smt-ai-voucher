import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# import plotly.express as px
import seaborn as sns

import urllib.request
import datetime
import json
import glob
import sys
import os

# from fbprophet import Prophet

import warnings
warnings.filterwarnings(action='ignore')


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.grid'] = False

pd.set_option('display.max_columns', 250)
pd.set_option('display.max_rows', 250)
pd.set_option('display.width', 100)

pd.options.display.float_format = '{:.2f}'.format





class NaverDataLabOpenAPI():
    """
    네이버 데이터랩 오픈 API 컨트롤러 클래스
    """


    def __init__(self, client_id, client_secret):
        """
        인증키 설정 및 검색어 그룹 초기화
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.keywordGroups = []
        self.url = "https://openapi.naver.com/v1/datalab/search"


    def add_keyword_groups(self, group_dict):
        """
        검색어 그룹 추가
        """

        keyword_gorup = {
            'groupName': group_dict['groupName'],
            'keywords': group_dict['keywords']
        }

        self.keywordGroups.append(keyword_gorup)
        print(f">>> Num of keywordGroups: {len(self.keywordGroups)}")


    def get_data(self, startDate, endDate, timeUnit, device, ages, gender):
        """
        요청 결과 반환
        timeUnit - 'date', 'week', 'month'
        device - None, 'pc', 'mo'
        ages = [], ['1' ~ '11']
        gender = None, 'm', 'f'
        """

        # Request body
        body = json.dumps({
            "startDate": startDate,
            "endDate": endDate,
            "timeUnit": timeUnit,
            "keywordGroups": self.keywordGroups,
            "device": device,
            "ages": ages,
            "gender": gender
        }, ensure_ascii=False)

        # Results
        request = urllib.request.Request(self.url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)
        request.add_header("Content-Type", "application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        rescode = response.getcode()
        if (rescode == 200):
            # Json Result
            result = json.loads(response.read())

            df = pd.DataFrame(result['results'][0]['data'])[['period']]
            for i in range(len(self.keywordGroups)):
                tmp = pd.DataFrame(result['results'][i]['data'])
                tmp = tmp.rename(columns={'ratio': result['results'][i]['title']})
                df = pd.merge(df, tmp, how='left', on=['period'])
            self.df = df.rename(columns={'period': '날짜'})
            self.df['날짜'] = pd.to_datetime(self.df['날짜'])

        else:
            print("Error Code:" + rescode)

        return self.df


    def plot_daily_trend(self):
        """
        일 별 검색어 트렌드 그래프 출력
        """
        colList = self.df.columns[1:]
        n_col = len(colList)

        fig = plt.figure(figsize=(12, 6))
        plt.title('일 별 검색어 트렌드', size=20, weight='bold')
        for i in range(n_col):
            sns.lineplot(x=self.df['날짜'], y=self.df[colList[i]], label=colList[i])
        plt.legend(loc='upper right')

        return fig


    def plot_monthly_trend(self):
        """
        월 별 검색어 트렌드 그래프 출력
        """
        df = self.df.copy()
        df_0 = df.groupby(by=[df['날짜'].dt.year, df['날짜'].dt.month]).mean().droplevel(0).reset_index().rename(
            columns={'날짜': '월'})
        df_1 = df.groupby(by=[df['날짜'].dt.year, df['날짜'].dt.month]).mean().droplevel(1).reset_index().rename(
            columns={'날짜': '년도'})

        df = pd.merge(df_1[['년도']], df_0, how='left', left_index=True, right_index=True)
        df['날짜'] = pd.to_datetime(df[['년도', '월']].assign(일=1).rename(columns={"년도": "year", "월": 'month', '일': 'day'}))

        colList = df.columns.drop(['날짜', '년도', '월'])
        n_col = len(colList)

        fig = plt.figure(figsize=(12, 6))
        plt.title('월 별 검색어 트렌드', size=20, weight='bold')
        for i in range(n_col):
            sns.lineplot(x=df['날짜'], y=df[colList[i]], label=colList[i])
        plt.legend(loc='upper right')

        return fig


    # def plot_pred_trend(self, days):
    #     """
    #     검색어 시계열 트렌드 예측 그래프 출력
    #     days: 예측일수
    #     """
    #     colList = self.df.columns[1:]
    #     n_col = len(colList)
    #
    #     fig_list = []
    #     for i in range(n_col):
    #         globals()[f"df_{str(i)}"] = self.df[['날짜', f'{colList[i]}']]
    #         globals()[f"df_{str(i)}"] = globals()[f"df_{str(i)}"].rename(columns={'날짜': 'ds', f'{colList[i]}': 'y'})
    #
    #         m = Prophet()
    #         m.fit(globals()[f"df_{str(i)}"])
    #
    #         future = m.make_future_dataframe(periods=days)
    #         forecast = m.predict(future)
    #         forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    #
    #         globals()[f"fig_{str(i)}"] = m.plot(forecast, figsize=(12, 6))
    #         plt.title(colList[i], size=20, weight='bold')
    #
    #         fig_list.append(globals()[f"fig_{str(i)}"])
    #
    #     return fig_list
if __name__ == '__main__':
    now = datetime.datetime.now()
    today_datetime = now.strftime("%Y%m%d%H%M%S")[2:]


    with open("config") as f:
        config = json.load(f)

    cid = config['client_id']
    cpwd = config['client_secret']

    naver = NaverDataLabOpenAPI(cid, cpwd)

    keyword_group_set = {
        'keyword_group_1': {'groupName': "마스크", 'keywords': ["마스크", "mask"]},
        'keyword_group_2': {'groupName': "핸드크림", 'keywords': ["핸드크림", "handcream"]},
        'keyword_group_3': {'groupName': "립스틱", 'keywords': ["립스틱", "Lipstick"]}
    }

    naver.add_keyword_groups(keyword_group_set['keyword_group_1'])
    naver.add_keyword_groups(keyword_group_set['keyword_group_2'])
    naver.add_keyword_groups(keyword_group_set['keyword_group_3'])


    startDate = "2020-01-01"
    endDate = "2020-12-31"
    timeUnit = 'month'
    device = ''
    ages = []
    gender = ''

    result = naver.get_data(startDate , endDate, timeUnit, device , ages , gender)

    print(result)

    result.to_csv(f'naver_trends_{startDate}_{endDate}_{today_datetime}.csv' , encoding='CP949')


    pass
