import os
import sys
import urllib.request
import json
import pandas as pd


def add_keyword_groups(group_dict):
    keywordGroups = []
    """
    검색어 그룹 추가
    """

    keyword_gorup = {
        "groupName": group_dict["groupName"],
        "keywords": group_dict["keywords"],
    }

    keywordGroups.append(keyword_gorup)
    print(f">>> Num of keywordGroups: {len(keywordGroups)}")
    return keywordGroups


dw = pd.DataFrame(
    [[20, 30, {"ab": "1", "we": "2", "as": "3"}, "String"]],
    columns=["ColA", "ColB", "ColC", "ColdD"],
)
#
print(dw)
print(dw.explode("ColC"))
# print(dw['ColC'].apply(pd.Series))

# df = pd.read_csv('naver_trend_search_2202240920.csv', encoding='CP949')
#
# print(df.dtypes)
# print(df.explode('data'))

# print(df['data'].apply(pd.Series))


sys.exit(0)
client_id = "7xfu_oeyiZjmiSKvG1kC"
client_secret = "puSsIc93xj"

url = "https://openapi.naver.com/v1/datalab/search"
body = (
    '{"startDate":"2017-01-01",'
    '"endDate":"2021-12-31",'
    '"timeUnit":"month",'
    '"keywordGroups":[{"groupName":"마스크","keywords":["마스크","mask"]},'
    '{"groupName":"핸드크림","keywords":["핸드크림","handcream"]}],'
    '"device":"pc",'
    '"ages":["1","2"],'
    '"gender":"f"}'
)


request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
request.add_header("Content-Type", "application/json")
response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()
if rescode == 200:
    response_body = response.read().decode("utf-8")
    raw = json.loads(response_body)
    result = json.loads(response.read())

    df = pd.DataFrame(result["results"][0]["data"])[["period"]]

    for i in range(len(self.keywordGroups)):
        tmp = pd.DataFrame(result["results"][i]["data"])
        tmp = tmp.rename(columns={"ratio": result["results"][i]["title"]})
        df = pd.merge(df, tmp, how="left", on=["period"])
    self.df = df.rename(columns={"period": "날짜"})
    self.df["날짜"] = pd.to_datetime(self.df["날짜"])
    data.to_csv("naver_trend_search_2202240920.csv", encoding="CP949", index=False)
    #
    # print(response_body.decode("utf-8"))
else:
    print("Error Code:" + rescode)
