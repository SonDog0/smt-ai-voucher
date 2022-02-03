import collections
import datetime
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
import nltk
from nltk.corpus import stopwords
import math

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
    print(kan_asin.kan_code.value_counts().keys())
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

    print(result.kan_code.value_counts().keys())

    return result


def tagging(lines, tag):
    if tag == 'NOUN':
        is_noun = lambda pos: pos == 'NOUN'

    elif tag == 'VERB':
        is_noun = lambda pos: pos == 'VERB'

    elif tag == 'ADJ':
        is_noun = lambda pos: pos == 'ADJ'

    elif tag == 'ALL':
        is_noun = lambda pos: pos == 'ADV' or pos == 'ADJ' or pos == 'VERB'

    elif tag == 'NOUN_VERB':
        is_noun = lambda pos: pos == 'NOUN' or pos == 'VERB'

    elif tag == 'ADJ_ADV':
        is_noun = lambda pos: pos == 'ADV' or pos == 'ADJ'

    elif tag == 'ADV':
        is_noun = lambda pos: pos == 'ADV'

    else:
        raise Exception('inccorect TAG name')

    nouns = [word for (word, pos) in nltk.pos_tag(lines, tagset='universal') if is_noun(pos)]
    return nouns

def udf_list_lower(list):
    return [x.lower() for x in list]


def is_in_word(lists, word):
    cnt = 0
    for item in lists:
        if item == word:
            cnt +=1
    return cnt

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return math.log(N/(df + 1))+1

def tfidf(t, d):
    return tf(t,d)* idf(t)


def get_ratio(p1, p2):
    return p1 / (p1 + p2)

if __name__ == "__main__":
    df = join_df()
    print(df.kan_code.value_counts().keys())
    sys.exit(0)
    df['review'] = df['review'].str.replace("[^a-zA-Z ]", "")
    df['review'].replace('', np.nan, inplace=True)
    df = df.dropna(how='any')  # Null 값 제거


    df['review'] = df['review'].map(lambda x: str(x).lower())

    df['tokenized'] = df['review'].apply(nltk.word_tokenize)

    # tag_name = 'NOUN'
    # tag_name = 'VERB'
    # tag_name = 'ADJ'
    tag_name = 'ADJ_ADV'
    df['tokenized'] = df['tokenized'].apply(tagging, args=[tag_name])

    df['tokenized'] = df['tokenized'].apply(udf_list_lower)

    df = df[df['tokenized'] != '[]']

    df['tokenized'] = df['tokenized'].apply(
        lambda x: x.replace('[', '').replace(']', '').replace(' ', '').replace("'", '').split(','))

    pos_series = df[df['label'] == 'positive'].tokenized
    neg_series = df[df['label'] == 'negative'].tokenized

    pos_list = [element.lower() for list_ in pos_series.values for element in list_]
    neg_list = [element.lower() for list_ in neg_series.values for element in list_]

    stop_words = stopwords.words('english')

    shopee_main = pd.read_excel('data/KAN상품분류_화장품매핑_210826.xlsx', sheet_name='Main')

    en_name = shopee_main['영문 키워드'].dropna().tolist()

    cos_stopword = [nltk.word_tokenize(result) for result in en_name]

    cos_stopword = list(set([item for sublist in cos_stopword for item in sublist]))

    cos_stopword = list(set(cos_stopword))

    stop_words = stop_words + cos_stopword

    pos_list_stopword = []
    neg_list_stopword = []

    for w in pos_list:
        if w not in stop_words:
            pos_list_stopword.append(w)

    for w in neg_list:
        if w not in stop_words:
            neg_list_stopword.append(w)

    top500_pos_word = list(dict(collections.Counter(pos_list_stopword).most_common()).keys())[:500]
    top500_neg_word = list(dict(collections.Counter(neg_list_stopword).most_common()).keys())[:500]

    for word in top500_pos_word:
        df[word] = df['tokenized'].apply(is_in_word, args=[word])

    for word in top500_neg_word:
        df[word] = df['tokenized'].apply(is_in_word, args=[word])

    df = df[df['review'].notna()]

    words_list = ['top30_neg_word', 'top30_pos_word']
    tfidf_ = []

    for w in words_list:

        docs = []

        neg_review_string = ",".join(df[df['label'] == 'negative']['review'].values.tolist())
        pos_review_string = ",".join(df[df['label'] == 'positive']['review'].values.tolist())

        docs.append(neg_review_string)
        docs.append(pos_review_string)

        vocab = eval(w)
        #     vocab.sort()

        N = len(docs)  # 총 문서의 수

        result = []
        for i in range(N):
            result.append([])
            d = docs[i]
            for j in range(len(vocab)):
                t = vocab[j]

                result[-1].append(tfidf(t, d))

        tfidf_.append(pd.DataFrame(result, columns=vocab, index=['NEG_DOC', 'POS_DOC']))

    tfidf_0 = tfidf_[0].T
    tfidf_1 = tfidf_[1].T

    now = datetime.now()
    today_datetime = now.strftime("%Y%m%d%H%M%S")[2:]

    tfidf_0['NEG_DOC_RATIO'] = tfidf_0.apply(lambda x: get_ratio(x.NEG_DOC, x.POS_DOC), axis=1)
    tfidf_0['POS_DOC_RATIO'] = tfidf_0.apply(lambda x: get_ratio(x.POS_DOC, x.NEG_DOC), axis=1)

    tfidf_1['NEG_DOC_RATIO'] = tfidf_1.apply(lambda x: get_ratio(x.NEG_DOC, x.POS_DOC), axis=1)
    tfidf_1['POS_DOC_RATIO'] = tfidf_1.apply(lambda x: get_ratio(x.POS_DOC, x.NEG_DOC), axis=1)

    tfidf_0.to_csv(f'NEG_WORD_500_TF-IDF_{today_datetime}_{tag_name}.csv', encoding='CP949')

    tfidf_1.to_csv(f'POS_WORD_500_TF-IDF_{today_datetime}_{tag_name}.csv', encoding='CP949')

    pass
