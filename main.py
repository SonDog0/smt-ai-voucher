# Importing the library

import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime

import xgboost
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


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

from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

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

col_list = [
    "roll_ATTITUDE",
    "pitch_ATTITUDE",
    "yaw_ATTITUDE",
    "error_rp_AHRS",
    "error_yaw_AHRS",
    "omegaIx_AHRS",
    "omegaIy_AHRS",
    "omegaIz_AHRS",
    "roll_AHRS2",
    "pitch_AHRS2",
    "yaw_AHRS2",
    "roll_AHRS3",
    "pitch_AHRS3",
    "yaw_AHRS3",
    "velocity_variance_EKF_STATUS_REPORT",
    "pos_horiz_variance_EKF_STATUS_REPORT",
    "pos_vert_variance_EKF_STATUS_REPORT",
    "compass_variance_EKF_STATUS_REPORT",
    "nav_roll_NAV_CONTROLLER_OUTPUT",
    "nav_pitch_NAV_CONTROLLER_OUTPUT",
    "xtrack_error_NAV_CONTROLLER_OUTPUT",
    "aspd_error_NAV_CONTROLLER_OUTPUT",
    "xacc_RAW_IMU",
    "yacc_RAW_IMU",
    "zacc_RAW_IMU",
    "xgyro_RAW_IMU",
    "ygyro_RAW_IMU",
    "zgyro_RAW_IMU",
    "xmag_RAW_IMU",
    "ymag_RAW_IMU",
    "zmag_RAW_IMU",
    "xacc_SCALED_IMU2",
    "yacc_SCALED_IMU2",
    "zacc_SCALED_IMU2",
    "xgyro_SCALED_IMU2",
    "ygyro_SCALED_IMU2",
    "zgyro_SCALED_IMU2",
    "xmag_SCALED_IMU2",
    "ymag_SCALED_IMU2",
    "zmag_SCALED_IMU2",
    "servo1_raw_SERVO_OUTPUT_RAW",
    "servo2_raw_SERVO_OUTPUT_RAW",
    "servo3_raw_SERVO_OUTPUT_RAW",
    "servo4_raw_SERVO_OUTPUT_RAW",
    "servo5_raw_SERVO_OUTPUT_RAW",
    "servo6_raw_SERVO_OUTPUT_RAW",
    "servo7_raw_SERVO_OUTPUT_RAW",
    "servo8_raw_SERVO_OUTPUT_RAW",
    "vibration_x_VIBRATION",
    "vibration_y_VIBRATION",
    "vibration_z_VIBRATION",
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


def make_output_data(result: pd.DataFrame, model, target, path):
    dir_name = "result/220111/cross_test/data/"
    file_name = f"SMT_out_{model}_{target}_rfe_std_cross_220113"
    result.to_csv(dir_name + file_name + ".csv", encoding="CP949", index=False)
    pass


def make_output_metric(y_true, y_pred):
    mae, mse, rmse, r2 = evaluation(y_true, y_pred)
    return


def fix_columns(df: pd.DataFrame):
    df["roll_AHRS_DIFF"] = df["roll_AHRS2"] - df["roll_AHRS3"]
    df["pitch_AHRS_DIFF"] = df["pitch_AHRS2"] - df["pitch_AHRS3"]
    df["yaw_AHRS_DIFF"] = df["yaw_AHRS2"] - df["yaw_AHRS3"]
    df["xacc_IMU_DIFF"] = df["xacc_RAW_IMU"] - df["xacc_SCALED_IMU2"]
    df["yacc_IMU_DIFF"] = df["yacc_RAW_IMU"] - df["yacc_SCALED_IMU2"]
    df["zacc_IMU_DIFF"] = df["zacc_RAW_IMU"] - df["zacc_SCALED_IMU2"]
    df["xgyro_IMU_DIFF"] = df["xgyro_RAW_IMU"] - df["xgyro_SCALED_IMU2"]
    df["ygyro_IMU_DIFF"] = df["ygyro_RAW_IMU"] - df["ygyro_SCALED_IMU2"]
    df["zgyro_IMU_DIFF"] = df["zgyro_RAW_IMU"] - df["zgyro_SCALED_IMU2"]
    df["xmag_IMU_DIFF"] = df["xmag_RAW_IMU"] - df["xmag_SCALED_IMU2"]
    df["ymag_IMU_DIFF"] = df["ymag_RAW_IMU"] - df["ymag_SCALED_IMU2"]
    df["zmag_IMU_DIFF"] = df["zmag_RAW_IMU"] - df["zmag_SCALED_IMU2"]

    df.drop(
        [
            "roll_AHRS2",
            "roll_AHRS3",
            "pitch_AHRS2",
            "pitch_AHRS3",
            "yaw_AHRS2",
            "yaw_AHRS3",
            "xacc_RAW_IMU",
            "xacc_SCALED_IMU2",
            "yacc_RAW_IMU",
            "yacc_SCALED_IMU2",
            "zacc_RAW_IMU",
            "zacc_SCALED_IMU2",
            "xgyro_RAW_IMU",
            "xgyro_SCALED_IMU2",
            "ygyro_SCALED_IMU2",
            "zgyro_RAW_IMU",
            "zgyro_SCALED_IMU2",
            "xmag_RAW_IMU",
            "xmag_SCALED_IMU2",
            "ymag_RAW_IMU",
            "ymag_SCALED_IMU2",
            "zmag_RAW_IMU",
            "zmag_SCALED_IMU2",
        ],
        inplace=True,
        axis=1,
    )

    return df


def modeling(
    which_model,
    X_train,
    y_train,
    X=None,
    y=None,
    is_cross=False,
):
    try:
        if which_model == "ridge":
            model = Ridge()

        elif which_model == "xgb":
            model = xgboost.XGBRegressor()

        elif which_model == "mlp":
            model = MLPRegressor()

        elif which_model == "svr":
            model = SVR()

        else:
            raise ValueError("check model name")

    except ValueError:
        print('ERROR!, check model name')


    """hyperparamerter tuning
    https://dining-developer.tistory.com/4
    """

    # predict by dataset
    """
    N-N(8:2) 
    N-F(ALL:ALL)
    """
    if is_cross == False:
        # split
        train_x, test_x, train_y, test_y = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        # fitting by split dataset
        model.fit(train_x, train_y)

        y_pred = model.predict(test_x)
        y_true = test_y
        result = test_x.copy()

    elif is_cross == True:
        # fitting by raw dataset
        model.fit(X_train, y_train)

        y_pred = model.predict(X)
        y_true = y
        result = X.copy()

    result[f"{target}_true"] = y_true
    result[f"{target}_pred"] = y_pred

    return result, y_true, y_pred


if __name__ == "__main__":

    model = "xgb"
    default_path = os.getcwd()  # /home/aiteam/son/pycharm
    result_path = default_path + '/result'

    sys.exit(0)

    output_metric = pd.DataFrame(columns=["TARGET", "MAE", "MSE", "RMSE", "R_SQUARE"])
    dicts = {}

    for tg in target_list:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"TARGET:{tg}, Start at {current_time}")
        target = tg

        # read_csv
        df = pd.read_csv("/home/aiteam/son/AI-voucher-Smatii/data/normal_dataset.csv")
        df_cross = pd.read_csv(
            "/home/aiteam/son/AI-voucher-Smatii/data/outlier_dataset.csv"
        )

        df = df[col_list]
        df = fix_columns(df)
        df_cross = df_cross[col_list]
        df_cross = fix_columns(df_cross)

        # make dataset
        x = df.drop(target, axis=1)
        y = df[target]
        x_cross = df_cross.drop(target, axis=1)
        y_cross = df_cross[target]

        # scailing X
        std_scaler = StandardScaler()
        std_scaler.fit(x)
        x_scale_trans = std_scaler.transform(x)
        scaled_x = pd.DataFrame(x_scale_trans, columns=x.columns)

        scaled_x_cross = pd.DataFrame(
            std_scaler.transform(x_cross), columns=x_cross.columns
        )

        # run model
        result, y_true, y_pred = modeling(
            X_train=scaled_x,
            y_train=y,
            X=scaled_x_cross,
            y=y_cross,
            is_cross=False,
            which_model="ridge",
        )

        # make output prediction
        make_output_data(result, "xgb", target)

        # eval & make output eval
        mae, mse, rmse, r2 = evaluation(y_true, y_pred)
        dicts.update(
            {"TARGET": target, "MAE": mae, "MSE": mse, "RMSE": rmse, "R_SQUARE": r2}
        )
        output_metric = output_metric.append(dicts, True)
        print(output_metric)

        pass

    output_metric.to_csv(
        "result/220111/cross_test/prediction/SMT_out_XGB_prediction_rfe_std_cross_220113.csv",
        encoding="CP949",
        index=False,
    )
