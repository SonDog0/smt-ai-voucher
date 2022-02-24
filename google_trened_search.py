import pytrends

from pytrends.request import TrendReq

pytrends = TrendReq(hl="en-US", tz=360)

kw_list = ["핸드크림"]
pytrends.build_payload(
    kw_list, cat=0, timeframe="2021-01-01 2021-12-31", geo="KR", gprop=""
)

#
print(pytrends.interest_over_time())
#
pytrends.interest_over_time().to_csv(
    "google_trends_handcream_190101-220201.csv", encoding="CP949"
)

# pytrends.get_historical_interest(kw_list, year_start=2018, month_start=1, day_start=1, hour_start=0, year_end=2018, month_end=2, day_end=1, hour_end=0, cat=0, geo='', gprop='', sleep=0).to_csv('google_trends_220224_by_hour.csv' , encoding='CP949')

print("checkOut")
