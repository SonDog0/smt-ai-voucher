import glob
import os
import shutil
import sys

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


tf.get_logger().setLevel("ERROR")


def calc_score(score):
    result = ""
    if score >= 0.5:
        result = "positive"
    else:
        result = "negative"

    return result


def review_analysis(model, df, review_col):
    review_list = df[review_col].tolist()
    result = tf.sigmoid(model(tf.constant(review_list)))
    result_list = [result[i][0].numpy() for i in range(len(result))]
    df["sentimental score"] = result_list
    df["label"] = df["sentimental score"].apply(calc_score)
    return df


def make_imdb_bert():

    reloaded_model = tf.saved_model.load("./imdb_bert")

    df = pd.read_csv(
        "data/review_all_211027.csv",
        encoding="utf-8-sig",
    )

    print(df)

    df = df[["id", "asin.original", "asin.variant", "rating", "review"]]
    # df = df[["id", "rating", "review"]]

    df = df[df["review"].notna()]

    from itertools import chain

    cols = df.columns.difference(["review"])
    review = df["review"].str.split(".")

    df = df.loc[df.index.repeat(review.str.len()), cols].assign(
        review=list(chain.from_iterable(review.tolist()))
    )

    df = df[df["review"].notna()]
    df = df[df["review"] != ""]

    step = 10000
    slice_num = 0
    cnt = 0
    itter = len(df)

    for i in range(10000, int(itter), step):
        cnt += 1
        print(i)
        slice_df = df.iloc[i - step : i]

        if i + step > int(itter):
            print("last")

            slice_df = df.iloc[i - step :]

        # print(f'i :{i}')
        # print(f'step : {step}')
        # print(f'df head : {slice_df.head(5)}')
        # print(f'df tail : {slice_df.tail(5)}')
        # print(f'len_df = {len(slice_df)}')

        df_sentimental = review_analysis(reloaded_model, slice_df, "review")
        df_sentimental.to_csv(
            f"./result/bert/review_all_sentimental_analysis_220124_{i}.csv",
            encoding="utf-8-sig",
            index=False,
        )


def concat_csv():
    appended_data = []
    for idx, infile in enumerate(
        glob.glob("/home/aiteam/son/pycharm/result/bert/*.csv")
    ):
        try:
            print(idx)
            data = pd.read_csv(infile, encoding="utf-8-sig")
            # store DataFrame in list
            # print(appended_data)
            appended_data.append(data)
        except:
            continue

    # see pd.concat documentation for more info
    appended_data = pd.concat(appended_data)

    print(len(appended_data))
    appended_data.to_csv(
        "/home/aiteam/son/pycharm/result/bert/amazon_sentimental_analysis_220203.csv",
        index=False,
        encoding="utf-8-sig",
    )


def join_df():
    kan_asin = pd.read_csv("data/KAN_AMAZON_IDMAP_202202031355.csv")
    df = pd.read_csv(
        "/home/aiteam/son/pycharm/result/bert/amazon_sentimental_analysis_220203.csv",
        encoding="utf-8-sig",
    )
    df.drop("asin.variant", axis=1, inplace=True)
    df.drop("id", axis=1, inplace=True)
    kan_asin.drop_duplicates("ASIN", inplace=True)

    result = pd.merge(
        df, kan_asin, left_on="asin.original", right_on="ASIN", how="inner"
    )

    return result


if __name__ == "__main__":
    df = join_df()

    pass
