# Importing the library

import os

import pandas as pd
import numpy as np
from datetime import datetime

import xgboost

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

target_list = [
    "roll_ATTITUDE",
    "pitch_ATTITUDE",
    "yaw_ATTITUDE",
    "xgyro_RAW_IMU",
    "ygyro_RAW_IMU",
    "zgyro_RAW_IMU",
    "xacc_RAW_IMU",
    "yacc_RAW_IMU",
    "zacc_RAW_IMU",
    "xmag_RAW_IMU",
    "ymag_RAW_IMU",
    "zmag_RAW_IMU",
    "vibration_x_VIBRATION",
    "vibration_y_VIBRATION",
    "vibration_z_VIBRATION",
]

feature_list = os.listdir(
    "/home/aiteam/son/AI-voucher-Smatii/result/FeatureSelection/220106"
)

def evaluation(y_hat, predictions):
    mae = mean_absolute_error(y_hat, predictions)
    mse = mean_squared_error(y_hat, predictions)
    rmse = np.sqrt(mean_squared_error(y_hat, predictions))
    r_squared = r2_score(y_hat, predictions)
    return mae, mse, rmse, r_squared


def feature_selection(x, y , k , sf = f_regression):
    skb = SelectKBest(score_func=sf, k=k)
    fit = skb.fit(x, y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    best_feature = list(featureScores.nlargest(k, 'Score').iloc[:,0])  # get k best columns to LIST

    return best_feature


def get_output(result:pd.DataFrame, model , target, k):
    dir_name = 'result/'
    file_name = f'SMATII_{model}_{target}_{k}'
    result.to_csv(dir_name + file_name , encoding='CP949', index=False)
    pass


def xgb(
    X_train,
    y_train,
    predict_all=False,
    X=None,
    y=None,
):

    """hyperparamerter tuning
    https://dining-developer.tistory.com/4
    """

    # make xgb model
    # xgb_model = xgboost.XGBRegressor(
    #     n_estimators=100,
    #     learning_rate=0.08,
    #     gamma=0,
    #     subsample=0.75,
    #     colsample_bytree=1,
    # )

    xgb_model = xgboost.XGBRegressor()
    param_grid = {'booster': ['gbtree'],
                  'max_depth': [5, 6, 8],
                  'min_child_weight': [1, 3, 5],
                  'gamma': [0, 1, 2, 3],
                  'nthread': [4],
                  'colsample_bytree': [0.5, 0.8],
                  'colsample_bylevel': [0.9],
                  'n_estimators': [50],
                  'random_state': [2]}

    gcv = GridSearchCV(xgb_model, param_grid=param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=4)

    # fitting
    gcv.fit(X_train, y_train)

    print('best params', gcv.best_params_)  # best param
    print('best score', gcv.best_score_)  # best score


    # predict by dataset
    if predict_all == False:
        y_pred = cross_val_predict(gcv, X_train, y_train, cv=5)
        y_true = y_train
        result = X_train.copy()

    elif predict_all == True:
        y_pred = cross_val_predict(gcv, X, y, cv=5)
        y_true = y
        result = X.copy()


    result[f"{target}_true"] = y_true
    result[f"{target}pred"] = y_pred

    return result


if __name__ == "__main__":

    # read_csv
    df = pd.read_csv('/home/aiteam/son/AI-voucher-Smatii/data/normal_dataset.csv')

    # set target col
    target = target_list[0]

    # make dataset
    x = df.drop(target , axis = 1)
    y = df[target]

    # feature selection
    cols = feature_selection(x=x , y=y , k=10)
    x = x[cols]
    print(x)

    # run xgb
    """
    X_train
    y_train
    predict_all=False
    X=None
    y=None"""

    result = xgb(X_train=x , y_train=y)
    get_output(result)

    print(result)

    pass
