import sys

import pandas as pd
import numpy as np

import xgboost
from imblearn.over_sampling import ADASYN

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

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

def make_oversampling_dataset(X,y):

    X_train_over, y_train_over = oversampling(X , y)
    # print(X_train_over)
    # print(y_train_over)
    #
    # print(y_train_over.value_counts())

    raw_data_oversampling = X_train_over.copy()
    raw_data_oversampling['label'] = y_train_over
    raw_data_oversampling.to_csv('raw_data_oversampling.csv' , index =False)

    # X_train_over.to_csv('220314test1.csv')
    # y_train_over.to_csv('220314test2.csv')


    return X_train_over , y_train_over

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
            # "roll_AHRS3",
            # "pitch_AHRS3",
            # "yaw_AHRS3",
            # # EKF_STATUS_REPORT
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
            # IMU
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
            # SCALED_PRESSURE
            "press_abs_SCALED_PRESSURE",
            "temperature_SCALED_PRESSURE"

        ]
    ]

    return df


def preprocessing_dataframe(df):
    # select HEXA Drone
    df = df[df["type_HEARTBEAT"] == 13]
    # sys.exit(0)
    #
    # df["roll_AHRS_DIFF"] = df["roll_AHRS2"] - df["roll_AHRS3"]
    # df["pitch_AHRS_DIFF"] = df["pitch_AHRS2"] - df["pitch_AHRS3"]
    # df["yaw_AHRS_DIFF"] = df["yaw_AHRS2"] - df["yaw_AHRS3"]
    #
    # df.drop(
    #     [
    #         # "roll_AHRS2",
    #         # "pitch_AHRS2",
    #         # "yaw_AHRS2",
    #         # "roll_AHRS3",
    #         # "pitch_AHRS3",
    #         # "yaw_AHRS3",
    #         # "type_HEARTBEAT"
    #     ],
    #     axis=1,
    #     inplace=True,
    # )

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
    from collections import Counter
    from imblearn.over_sampling import SMOTE

    # adasyn = ADASYN(random_state=22)
    # X_train_over, y_train_over = adasyn.fit_resample(train_X, train_y)

    smote = SMOTE(random_state=0)

    X_train_over, y_train_over = smote.fit_resample(train_X, train_y)
    print(Counter(y_train_over))

    return X_train_over, y_train_over


def modeling_randomforest(train_X, train_y):
    model = RandomForestClassifier()
    model.fit(train_X, train_y)

    # # Number of trees in random forest
    # n_estimators = [100]
    # # Number of features to consider at every split
    # max_features = ["auto", "sqrt"]
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    #
    # random_grid = {
    #     "n_estimators": n_estimators,
    #     "max_features": max_features,
    #     "max_depth": max_depth,
    #     "min_samples_split": min_samples_split,
    #     "min_samples_leaf": min_samples_leaf,
    #     "bootstrap": bootstrap,
    # }
    #
    #
    #
    # # Use the random grid to search for best hyperparameters
    # # First create the base model to tune
    # rf = RandomForestClassifier()
    # # Random search of parameters, using 3 fold cross validation,
    # # search across 100 different combinations, and use all available cores
    # model = RandomizedSearchCV(
    #     estimator=rf,
    #     param_distributions=random_grid,
    #     n_iter=100,
    #     cv=3,
    #     verbose=2,
    #     random_state=42,
    #     n_jobs=-1,
    # )
    # # Fit the random search model
    # # model.fit(X_train_over, y_train_over)
    #
    # # model = xgboost.XGBClassifier()
    #
    # model.fit(train_X, train_y)

    return model


def modeling_XGBoost(train_X , train_y):


    # from xgboost import XGBClassifier
    #
    # params = {
    #     'objective': 'binary:logistic',
    #     'max_depth': 4,
    #     'alpha': 10,
    #     'learning_rate': 1.0,
    #     'n_estimators': 100
    # }
    #
    # model = XGBClassifier(**params)
    # model.fit(train_X, train_y)
    #
    # scores = cross_val_score(xgb_clf, train_X, train_y, scoring='f1')
    #
    # print(scores)
    #
    # sys.exit(0)
    #
    from xgboost import cv


    # # data_dmatrix = xgboost.DMatrix(data=train_X, label=train_y)
    # params = {'objective': 'binary:logistic', 'eval_metric': 'logloss',
    #           'eta': 0.01,
    #           'subsample': 0.1}
    # xgb_cv = xgboost.cv(dtrain=data_dmatrix, params=params, nfold=5, metrics='logloss', seed=42)
    #
    model = xgboost.XGBClassifier()

    #
    # from sklearn.tree import DecisionTreeClassifier
    #
    # dt = DecisionTreeClassifier()
    #
    # from sklearn.model_selection import GridSearchCV
    #
    # params = {
    #     'max_depth': range(1,10),
    #     'min_samples_leaf': range(1,5),
    #     'criterion': ["gini", "entropy"]
    # }
    #
    # model = RandomizedSearchCV(
    #     estimator=dt,
    #     param_distributions=params,
    #     cv=3,
    #     verbose=2,
    #     random_state=42,
    #     n_jobs=-1,
    # )
    #
    #
    model.fit(train_X , train_y)

    # from sklearn.svm import SVC
    #
    #
    # from sklearn.model_selection import GridSearchCV
    #
    # # defining parameter range
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf']}
    #
    # model = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    # model.fit(train_X, train_y)


    # model.fit(train_X , train_y)

    # from sklearn.model_selection import cross_val_score
    # accuracies = cross_val_score(estimator=xgb_clf, X=train_X, y=train_y, cv=5)
    #
    # print(accuracies)
    # print(accuracies.mean())

    return model


def modeling_KNN(train_X, train_y):


    params = {'n_neighbors': [5, 7, 9],
              'leaf_size': [3],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto'],
              'n_jobs': [-1]}

    knn = KNeighborsClassifier(n_jobs=-1)

    model = RandomizedSearchCV(knn, param_distributions=params, n_jobs=-1, random_state=42, n_iter=10, cv=3)

    model.fit(train_X, train_y)

    return model



def prediction(model,model_name, test_X, test_y):

    y_pred = model.predict(test_X)

    print("Precision Score : {}".format(precision_score(test_y, y_pred)))
    print("Recall Score : {}".format(recall_score(test_y , y_pred)))
    print("Accuracy Score : {}".format(accuracy_score(test_y, y_pred)))
    print("F1 Score : {}".format(f1_score( test_y , y_pred)))

    print(confusion_matrix(test_y, y_pred))

    Precision = precision_score(test_y, y_pred)
    Recall = recall_score(test_y, y_pred)
    Accuracy = accuracy_score(test_y, y_pred)
    F1 = f1_score(test_y, y_pred)


    test_X["y_pred"] = y_pred
    test_X["y_true"] = test_y

    test_X.to_csv(f"result/{model_name}_220314.csv" , index = False)

    output_metric = pd.DataFrame(columns=["Precision", "Recall", "Accuracy", "F1"])

    dicts = {}

    dicts.update(
        {"Precision": Precision, "Recall": Recall, "Accuracy": Accuracy, "F1": F1}
    )
    output_metric = output_metric.append(dicts, True)

    output_metric.to_csv(
        f"result/{model_name}_220314_metrics.csv"
    )



if __name__ == "__main__":

    # show_dataset()

    df = pd.concat([normal, fault], keys=[0, 1])
    df = df.reset_index(level=1, drop=True).reset_index()
    df.rename(columns={"index": "label"}, inplace=True)
    df.drop(["filename"], axis=1, inplace=True)

    df = select_feature(df)
    df = preprocessing_dataframe(df)

    print(df)  # columns : 50
    # print(df.columns)
    # print(df["label"].value_counts())

    # sys.exit(0)

    X = df.drop("label", axis=1)
    y = df["label"]


    # X, y = make_oversampling_dataset(X,y)

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # 0 : 101260
    # 1 : 203

    # oversampling
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=0, sampling_strategy=0.5)
    X_train_over, y_train_over = smote.fit_resample(train_X, train_y)


    # make dataset
    train_dataset = X_train_over.copy()
    train_dataset['label'] = y_train_over
    train_dataset.to_csv(
        'training_dataset.csv' , index = False
    )

    test_dataset = test_X.copy()
    test_dataset['label'] = test_y
    test_dataset.to_csv(
        'test_dataset.csv' , index = False
    )

    raw_dataset = pd.concat([train_dataset , test_dataset])
    raw_dataset.to_csv(
        'raw_dataset.csv' ,index = False
    )

    # result = feature_selection(X, y, 5)
    # X = X[result]

    # XGB
    xgb_model = modeling_XGBoost(X_train_over, y_train_over)

    # selector = RFE(xgb_model, n_features_to_select=10, step=1)
    # selector = selector.fit(train_X, train_y)
    #
    # filter = selector.support_
    # ranking = selector.ranking_
    #
    # print("Mask data: ", filter)
    # print("Ranking: ", ranking)
    #
    # print(train_X.columns)
    # print(train_X.columns[filter])

    prediction(xgb_model, 'xgb', test_X, test_y)

    # KNN
    # knn_model = modeling_KNN(X_train_over, y_train_over)
    # # #
    # prediction(knn_model, 'knn', test_X, test_y)

    # RF
    # rf_model = modeling_randomforest(X_train_over, y_train_over)

    # prediction(rf_model, 'randomforest',test_X, test_y)




    # TODO : Classfication

    # TODO : Feature Importance

    # TODO : Oversampling
