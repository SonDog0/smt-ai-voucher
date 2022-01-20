# Importing the library

import os
import sys
from pathlib import Path
import pickle

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

# TODO get parameter from script & automatical running
# Define path
now = datetime.now()
# xgb , ridge, mlp , svr
model_name = "svr"
dtype = "N-N"

is_cross = False


default_path = os.getcwd()  # /home/aiteam/son/pycharm
result_path = default_path + "/result"  # /home/aiteam/son/pycharm/result
dir_name_ymd = now.strftime("%Y%m%d%H%M")[2:]  # 2201171200
dir_name = (
    result_path + "/" + dir_name_ymd + "/" + model_name
)  # # /home/aiteam/son/pycharm/result/220117/xgb

# make directory by Y-M-d
Path(dir_name).mkdir(parents=True, exist_ok=True)

# custom MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluation(y_hat, predictions):
    mape = mean_absolute_percentage_error(y_hat , predictions)
    mae = mean_absolute_error(y_hat, predictions)
    mse = mean_squared_error(y_hat, predictions)
    rmse = np.sqrt(mean_squared_error(y_hat, predictions))
    r_squared = r2_score(y_hat, predictions)
    return mape, mae, mse, rmse, r_squared


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


def make_output_data(result: pd.DataFrame, model, target, dtype):

    file_name = f"SMT_{dtype}_{model}_{target}.csv"

    result.to_csv(dir_name + "/" + file_name, encoding="CP949", index=False)

    return None


def make_output_metric(y_true, y_pred):
    mape , mae, mse, rmse, r2 = evaluation(y_true, y_pred)
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
            "pitch_ATTITUDE",
            "yaw_ATTITUDE",
            "xacc_RAW_IMU",
            "yacc_RAW_IMU",
            "zacc_RAW_IMU",
            "xgyro_RAW_IMU",
            "ygyro_RAW_IMU",
            "zgyro_RAW_IMU",
            "xmag_RAW_IMU",
            "ymag_RAW_IMU",
            "zmag_RAW_IMU",
            "roll_AHRS2",
            "roll_AHRS3",
            "pitch_AHRS2",
            "pitch_AHRS3",
            "yaw_AHRS2",
            "yaw_AHRS3",
            "xacc_SCALED_IMU2",
            "yacc_SCALED_IMU2",
            "zacc_SCALED_IMU2",
            "xgyro_SCALED_IMU2",
            "ygyro_SCALED_IMU2",
            "zgyro_SCALED_IMU2",
            "xmag_SCALED_IMU2",
            "ymag_SCALED_IMU2",
            "zmag_SCALED_IMU2",
        ],
        inplace=True,
        axis=1,
    )

    return df

def clf_modeling(
    which_model,
    normal_x,
    normal_y,
    fault_x,
    fault_y
):
    try:
        if which_model == "xgb":
            model = xgboost.XGBClassifier()
        else:
            raise ValueError()

    except ValueError:
        print("ERROR!, check model name")


    # 합쳐
    X = pd.concat([normal_x, fault_x]).reset_index()
    X = X.drop('index', axis=1)
    y = pd.concat([normal_y, fault_y], keys=[0,1])
    y = y.reset_index(level=1 , drop=True).reset_index()
    y.rename(columns={'index' : 'label'} , inplace = True)


    X['roll_ATTITUDE'] = y['roll_ATTITUDE']
    X['label'] = y['label']

    x_data = X.drop('label' , axis=1)
    y_data = X['label']

    train_X, test_X, train_y, test_y = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    print(y_data)
    # # oversampling
    # print(X_train)
    # print(y_train)

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=0)
    X_train_over, y_train_over = smote.fit_sample(train_X, train_y)


    #
    # print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)
    # print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
    # print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
    #

    # modeling

    model.fit(X_train_over , y_train_over)

    y_pred = model.predict(test_X)

    print('Precision Score : {}'.format(precision_score(y_pred, test_y)))
    print('Recall Score : {}'.format(recall_score(y_pred, test_y)))
    print('Accuracy Score : {}'.format(accuracy_score(y_pred, test_y)))
    print('F1 Score : {}'.format(f1_score(y_pred, test_y)))

    r = pd.DataFrame()
    r['y_true'] = test_y
    r['y_pred'] = y_pred


    pickle.dump(model, open('220119model_3', 'wb'))



    sys.exit(0)

    # eval




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

        elif which_model == "xgb_c":
            model = xgboost.XGBClassifier()
            is_cross = True

        else:
            raise ValueError()

    except ValueError:
        print("ERROR!, check model name")

    """hyperparamerter tuning
    https://dining-developer.tistory.com/4
    """

    # predict by dataset
    """
    N-N(8:2) 
    N-F(ALL:ALL)
    """

    if is_cross == False:

        if X_train["servo5_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X_train.drop("servo5_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
        if X_train["servo6_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X_train.drop("servo6_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
        if X_train["servo7_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X_train.drop("servo7_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
        if X_train["servo8_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X_train.drop("servo8_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)

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

        if X["servo5_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X.drop("servo5_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
            X_train.drop("servo5_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
        if X["servo6_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X.drop("servo6_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
            X_train.drop("servo6_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
        if X["servo7_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X.drop("servo7_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
            X_train.drop("servo7_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
        if X["servo8_raw_SERVO_OUTPUT_RAW"].sum() == 0:
            X.drop("servo8_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)
            X_train.drop("servo8_raw_SERVO_OUTPUT_RAW", inplace=True, axis=1)

        # fitting by raw dataset
        model.fit(X_train, y_train)

        y_pred = model.predict(X)
        y_true = y
        result = X.copy()

    result[f"{target}_true"] = y_true
    result[f"{target}_pred"] = y_pred

    return result, y_true, y_pred


if __name__ == "__main__":

    output_metric = pd.DataFrame(columns=["TARGET", "MAPE", "MAE", "MSE", "RMSE", "R_SQUARE"])
    dicts = {}

    for target in target_list:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"TARGET:{target}, Start at {current_time}")

        # read_csv
        df = pd.read_csv("/home/aiteam/son/AI-voucher-Smatii/data/normal_dataset.csv")
        df_cross = pd.read_csv(
            "/home/aiteam/son/AI-voucher-Smatii/data/outlier_dataset.csv"
        )

        # set column
        df = df[col_list]
        df = fix_columns(df)
        df_cross = df_cross[col_list]
        df_cross = fix_columns(df_cross)

        # set row, only hexa drone
        df = df[df["servo5_raw_SERVO_OUTPUT_RAW"] != 0]
        df = df[df["servo6_raw_SERVO_OUTPUT_RAW"] != 0]

        df_cross = df_cross[df_cross["servo5_raw_SERVO_OUTPUT_RAW"] != 0]
        df_cross = df_cross[df_cross["servo6_raw_SERVO_OUTPUT_RAW"] != 0]

        # make dataset X, y
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

        clf_modeling(which_model='xgb' , normal_x=scaled_x , normal_y=y , fault_x=scaled_x_cross , fault_y=y_cross)


        # run model
        result, y_true, y_pred = modeling(
            X_train=scaled_x,
            y_train=y,
            X=scaled_x_cross,
            y=y_cross,
            is_cross=is_cross,
            which_model=model_name,
        )

        # make output prediction
        make_output_data(result=result, model=model_name, target=target, dtype=dtype)

        # eval & make output eval
        mape, mae, mse, rmse, r2 = evaluation(y_true, y_pred)
        dicts.update(
            {"TARGET": target, "MAPE" : mape , "MAE": mae, "MSE": mse, "RMSE": rmse, "R_SQUARE": r2}
        )
        output_metric = output_metric.append(dicts, True)
        print(output_metric)

    prediction_file_name = f"SMT_{dtype}_{model_name}_prediction.csv"
    output_metric.to_csv(
        dir_name + "/" + prediction_file_name,
        encoding="CP949",
        index=False,
    )
