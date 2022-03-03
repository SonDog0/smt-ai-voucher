import sys

import pandas as pd
import glob
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


NEG_PATH = "result/bert_kancode/negative/"
POS_PATH = "result/bert_kancode/positive/"

neg_series = pd.Series()
pos_series = pd.Series()


def make_heat_map_csv():

    neg_file_list = os.listdir(NEG_PATH)
    pos_file_list = os.listdir(POS_PATH)

    df_kan = pd.read_excel("data/KAN상품분류_화장품매핑_220208.xlsx", sheet_name="Main")
    kan_list = df_kan["영문 키워드"].dropna().tolist()
    df_kan_dummy = pd.DataFrame(columns=["w1", "w2"])

    pos_300_list = pd.read_csv("data/amazon_bert_kcode_positive_220222_300.csv").iloc[
        :, 0
    ]
    neg_300_list = pd.read_csv("data/amazon_bert_kcode_negative_220222_300.csv").iloc[
        :, 0
    ]

    print(neg_300_list)

    for w1 in kan_list:
        for w2 in kan_list:
            df_kan_dummy.loc[len(df_kan_dummy)] = [w1, w2]
    print("finish make kancode dummy dataframe")

    print(df_kan_dummy)

    count_list = []
    for i in range(0, len(df_kan_dummy)):
        w1 = df_kan_dummy.iloc[i, 0]
        w2 = df_kan_dummy.iloc[i, 1]

        w1_csv = [t for t in neg_file_list if w1 in t][0]
        w2_csv = [t for t in neg_file_list if w2 in t][0]

        print(w1_csv)
        print(w2_csv)

        w1_df_list = pd.read_csv(NEG_PATH + w1_csv).iloc[:, 0]
        w2_df_list = pd.read_csv(NEG_PATH + w2_csv).iloc[:, 0]

        w1_w2_inter = set.intersection(set(w1_df_list), set(w2_df_list))
        inter_length = len(set.intersection(set(w1_w2_inter), set(neg_300_list)))

        count_list.append(inter_length)

    df_kan_dummy["count"] = count_list

    df_kan_dummy.to_csv("heatmap_test_220222.csv", index=False)


if __name__ == "__main__":
    # make_heat_map_csv()
    plt.close("all")
    plt.rcParams["figure.figsize"] = [48, 32]

    raw_df = pd.read_csv("heatmap_test_220222.csv")

    df = raw_df.pivot("w1", "w2", "count")

    print(df)

    sns.heatmap(df, annot=True, fmt="d", xticklabels=True, yticklabels=True)

    plt.title("kan code heatmap", fontsize=20)

    plt.savefig("kan_code_headmap.png", dpi=400)

    plt.show()

    sys.exit(0)

    neg_file_list = os.listdir(NEG_PATH)

    print(neg_file_list)
    print(len(neg_file_list))

    pos_file_list = os.listdir(POS_PATH)

    print(pos_file_list)
    print(len(pos_file_list))

    for neg, pos in zip(neg_file_list, pos_file_list):
        neg_df = pd.read_csv(NEG_PATH + neg, encoding="utf-8-sig").iloc[:, 0]
        pos_df = pd.read_csv(POS_PATH + pos, encoding="utf-8-sig").iloc[:, 0]
        neg_series = neg_series.append(neg_df)
        pos_series = pos_series.append(pos_df)

        # print(neg_df)
        # print(type(neg_df))
        # print(pos_df)

    # print(pos_series.value_counts()[:])
    # neg_series.value_counts()[:300].to_csv('data/amazon_bert_kcode_negative_220222_300.csv')
    # pos_series.value_counts()[:300].to_csv('data/amazon_bert_kcode_positive_220222_300.csv')

    # print(neg_series.value_counts()[:50])
    # print(len(neg_series))

    plt.rcParams["figure.figsize"] = [24, 18]
    flights = sns.load_dataset("flights")

    print(flights)

    df = flights.pivot("month", "year", "passengers")

    print(df)

    svm = sns.heatmap(df, annot=True, fmt="d")

    plt.title("Annoteat cell with numeric value", fontsize=20)

    plt.show()

    # sns.heatmap(df, annot=True, fmt='d')

    # plt.title('Annoteat cell with numeric value', fontsize=20)

    # plt.show()

    pass
