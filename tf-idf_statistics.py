import sys

import pandas as pd
import glob
import os
import datetime

NEG_PATH = "result/bert_kancode/negative/"
POS_PATH = "result/bert_kancode/positive/"

neg_series = pd.Series()

if __name__ == '__main__':

    neg_file_list  = os.listdir(NEG_PATH)

    print(neg_file_list)
    print(len(neg_file_list))

    pos_file_list = os.listdir(POS_PATH)

    print(pos_file_list)
    print(len(pos_file_list))

    for neg, pos in zip(neg_file_list, pos_file_list):
        neg_df = pd.read_csv(NEG_PATH + neg , encoding='utf-8-sig').iloc[:50, 0]
        pos_df = pd.read_csv(POS_PATH + pos, encoding='utf-8-sig').iloc[:50, 0]
        neg_series = neg_series.append(neg_df)


        # print(neg_df)
        # print(type(neg_df))
        # print(pos_df)

    print(neg_series.value_counts())
    print(len(neg_series))

    pass