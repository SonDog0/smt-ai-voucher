# Importing the library

import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime

import xgboost

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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
    "vibration_z_VIBRATION"
]

def evaluation(y_hat, predictions):
    mae = mean_absolute_error(y_hat, predictions)
    mse = mean_squared_error(y_hat, predictions)
    rmse = np.sqrt(mean_squared_error(y_hat, predictions))
    r_squared = r2_score(y_hat, predictions)
    return mae, mse, rmse, r_squared


def feature_selection(x, y, k, sf=f_regression):
    skb = SelectKBest(score_func=sf, k=k)
    fit = skb.fit(x, y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]  # naming the dataframe columns
    best_feature = list(
        featureScores.nlargest(k, "Score").iloc[:, 0]
    )  # get k best columns to LIST

    return best_feature


def make_output_data(result: pd.DataFrame, model, target):
    dir_name = "result/"
    file_name = f"SMT_out_{model}_{target}_mm_220111"
    result.to_csv(dir_name + file_name + ".csv", encoding="CP949", index=False)
    pass


def make_output_metric(y_true, y_pred):
    mae, mse, rmse, r2 = evaluation(y_true, y_pred)
    return


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

    param_grid = {
        "booster": ["gbtree"],
        "max_depth": [5, 6, 8],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 1, 2, 3],
        "nthread": [4],
        "colsample_bytree": [0.5, 0.8],
        "colsample_bylevel": [0.9],
        "n_estimators": [50],
        "random_state": [2],
    }

    # gcv = GridSearchCV(
    #     xgb_model,
    #     cv=5,
    #     scoring="neg_root_mean_squared_error",
    #     n_jobs=4,
    # )

    # split
    train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # fitting
    xgb_model.fit(train_x, train_y)
    #
    # print("best params", gcv.best_params_)  # best param
    # print("best score", gcv.best_score_)  # best score

    # predict by dataset
    if predict_all == False:
        y_pred = cross_val_predict(xgb_model, X_train, y_train, cv=5)
        y_true = y_train
        result = X_train.copy()

    elif predict_all == True:
        y_pred = cross_val_predict(xgb_model, X, y, cv=5)
        y_true = y
        result = X.copy()

    result[f"{target}_true"] = y_true
    result[f"{target}_pred"] = y_pred

    return result, y_true, y_pred


if __name__ == "__main__":

    model = "xgb"

    output_metric = pd.DataFrame(
        columns=["TARGET", "MAE", "MSE", "RMSE", "R_SQUARE"]
    )
    dicts = {}

    for tg in target_list:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"TARGET:{tg}, Start at {current_time}")
        target = tg

        # read_csv
        df = pd.read_csv(
            "/home/aiteam/son/AI-voucher-Smatii/data/outlier_dataset.csv"
        )

        # make dataset
        x = df.drop(target, axis=1)
        y = df[target]

        std_scaler = MinMaxScaler()
        std_scaler.fit(x)

        x_scale_trans = std_scaler.transform(x)

        scaled_x = pd.DataFrame(x_scale_trans, columns=x.columns)



        # feature selection
        # cols = feature_selection(x=x, y=y, k=k)
        # x = x[cols]

        # run model
        result, y_true, y_pred = xgb(X_train=scaled_x, y_train=y)

        # make output prediction
        make_output_data(result, "xgb", target)

        # eval & make output eval
        mae, mse, rmse, r2 = evaluation(y_true, y_pred)
        dicts.update(
            {
                "TARGET": target,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R_SQUARE": r2,
            }
        )
        output_metric = output_metric.append(dicts, True)
        print(output_metric)

        pass

    output_metric.to_csv("SMT_out_XGB_prediction_mm_220111.csv", encoding="CP949", index=False)
