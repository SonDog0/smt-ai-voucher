import sys

import pandas as pd
import numpy as np

import xgboost

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


normal = pd.read_csv("data/normal.csv")

fault = pd.read_csv("data/fault.csv")


def show_dataset():
    print(normal.head(5))
    print(normal.columns)
    print(len(normal))  # 135,631

    print("===============================")

    print(fault.head(5))
    print(fault.columns)
    print(len(fault))  # 251


def select_feature(df):

    df = df[
        [
            # ATTITUDE
            "roll_ATTITUDE",
            "pitch_ATTITUDE",
            "yaw_ATTITUDE",
            "rollspeed_ATTITUDE",
            "pitchspeed_ATTITUDE",
            "yawspeed_ATTITUDE",
            # AHRS
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
            # EKF_STATUS_REPORT
            "velocity_variance_EKF_STATUS_REPORT",
            "pos_horiz_variance_EKF_STATUS_REPORT",
            "pos_vert_variance_EKF_STATUS_REPORT",
            "compass_variance_EKF_STATUS_REPORT",
            # NAV_CONTROLLER_OUTPUT
            "xtrack_error_NAV_CONTROLLER_OUTPUT",
            "aspd_error_NAV_CONTROLLER_OUTPUT",
            # SERVO_OUTPUT_RAW
            "servo1_raw_SERVO_OUTPUT_RAW",
            "servo2_raw_SERVO_OUTPUT_RAW",
            "servo3_raw_SERVO_OUTPUT_RAW",
            "servo4_raw_SERVO_OUTPUT_RAW",
            "servo5_raw_SERVO_OUTPUT_RAW",
            "servo6_raw_SERVO_OUTPUT_RAW",
            # VIBRATION
            "vibration_x_VIBRATION",
            "vibration_y_VIBRATION",
            "vibration_z_VIBRATION",
            # HEARTBEAT
            "type_HEARTBEAT",
            # LABEL
            "label",
        ]
    ]

    return df


def preprocessing_dataframe(df):
    # select HEXA Drone
    df = df[df["type_HEARTBEAT"] == 13]
    print(df)
    print(df.columns)
    # sys.exit(0)

    df["roll_AHRS_DIFF"] = df["roll_AHRS2"] - df["roll_AHRS3"]
    df["pitch_AHRS_DIFF"] = df["pitch_AHRS2"] - df["pitch_AHRS3"]
    df["yaw_AHRS_DIFF"] = df["yaw_AHRS2"] - df["yaw_AHRS3"]

    df.drop(
        [
            "roll_AHRS2",
            "pitch_AHRS2",
            "yaw_AHRS2",
            "roll_AHRS3",
            "pitch_AHRS3",
            "yaw_AHRS3",
            "type_HEARTBEAT",
        ],
        axis=1,
        inplace=True,
    )

    return df


def feature_selection(x, y, k, sf=f_classif):
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


def oversampling(train_X, train_y):
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=0)
    smote.fit
    X_train_over, y_train_over = smote.fit_resample(train_X, train_y)

    return X_train_over, y_train_over


def modeling_randomforest(train_X, train_y):

    # Number of trees in random forest
    n_estimators = [100]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }



    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    model = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    # Fit the random search model
    # model.fit(X_train_over, y_train_over)

    # model = xgboost.XGBClassifier()

    model.fit(train_X, train_y)

    return model


def modeling_XGBoost(train_X , train_y):

    model = xgboost.XGBClassifier()

    model.fit(train_X , train_y)

    return model


def modeling_KNN(train_X, train_y):

    std_scaler = StandardScaler()
    std_scaler.fit(train_X)
    train_x_scaling = std_scaler.transform(train_X)

    params = {'n_neighbors': [5, 7, 9],
              'leaf_size': [3],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto'],
              'n_jobs': [-1]}

    knn = KNeighborsClassifier(n_jobs=-1)

    model = RandomizedSearchCV(knn, param_distributions=params, n_jobs=-1, random_state=42, n_iter=100, cv=3)

    model.fit(train_x_scaling, train_y)

    return model



def prediction(model,model_name, test_X, test_y):

    y_pred = model.predict(test_X)

    print("Precision Score : {}".format(precision_score(y_pred, test_y)))
    print("Recall Score : {}".format(recall_score(y_pred, test_y)))
    print("Accuracy Score : {}".format(accuracy_score(y_pred, test_y)))
    print("F1 Score : {}".format(f1_score(y_pred, test_y)))

    Precision = precision_score(y_pred, test_y)
    Recall = recall_score(y_pred, test_y)
    Accuracy = accuracy_score(y_pred, test_y)
    F1 = f1_score(y_pred, test_y)


    test_X["y_pred"] = y_pred
    test_X["y_true"] = test_y

    test_X.to_csv(f"result/{model_name}_220310.csv")

    output_metric = pd.DataFrame(columns=["Precision", "Recall", "Accuracy", "F1"])

    dicts = {}

    dicts.update(
        {"Precision": Precision, "Recall": Recall, "Accuracy": Accuracy, "F1": F1}
    )
    output_metric = output_metric.append(dicts, True)

    output_metric.to_csv(
        f"result/{model_name}_220310_metrics.csv"
    )



if __name__ == "__main__":

    # show_dataset()

    df = pd.concat([normal, fault], keys=[0, 1])
    df = df.reset_index(level=1, drop=True).reset_index()
    df.rename(columns={"index": "label"}, inplace=True)
    df.drop(["filename"], axis=1, inplace=True)

    df = select_feature(df)
    df = preprocessing_dataframe(df)

    print(df)  # columns : 30
    print(df.columns)
    print(df["label"].value_counts())

    # sys.exit(0)

    X = df.drop("label", axis=1)
    y = df["label"]

    # result = feature_selection(X, y, 10)
    # ['pitch_AHRS_DIFF', 'error_yaw_AHRS', 'pos_horiz_variance_EKF_STATUS_REPORT', 'error_rp_AHRS', 'servo5_raw_SERVO_OUTPUT_RAW', 'servo3_raw_SERVO_OUTPUT_RAW', 'servo1_raw_SERVO_OUTPUT_RAW', 'servo2_raw_SERVO_OUTPUT_RAW', 'velocity_variance_EKF_STATUS_REPORT', 'pos_vert_variance_EKF_STATUS_REPORT']
    # ['zgyro_SCALED_IMU2', 'pitch_AHRS_DIFF', 'error_yaw_AHRS', 'accel_cal_z_SENSOR_OFFSETS', 'pos_horiz_variance_EKF_STATUS_REPORT', 'mag_ofs_x_SENSOR_OFFSETS', 'chan5_raw_RC_CHANNELS', 'voltages1_BATTERY_STATUS', 'gyro_cal_z_SENSOR_OFFSETS', 'gyro_cal_y_SENSOR_OFFSETS']

    # X = X[result]

    # print(result)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    X_train_over, y_train_over = oversampling(train_X, train_y)
    #
    # print(len(X_train_over))
    # print(len(y_train_over))
    #
    print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', train_X.shape, train_y.shape)
    print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)
    print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())

    # XGB
    xgb_model = modeling_XGBoost(X_train_over, y_train_over)

    selector = RFE(xgb_model, n_features_to_select=10, step=1)
    selector = selector.fit(X_train_over, y_train_over)
    filter = selector.support_
    ranking = selector.ranking_

    print("Mask data: ", filter)
    print("Ranking: ", ranking)

    print(X_train_over.columns)
    print(X_train_over.columns[filter])


    #
    # prediction(xgb_model, 'xgb', test_X, test_y)


    # KNN
    # knn_model = modeling_KNN(X_train_over, y_train_over)
    #
    # prediction(knn_model, 'knn', test_X, test_y)

    # RF
    # rf_model = modeling_randomforest(X_train_over, y_train_over)
    #
    # prediction(rf_model, 'randomforest',test_X, test_y)




    # TODO : Classfication

    # TODO : Feature Importance

    # TODO : Oversampling
