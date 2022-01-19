import sys

import pandas as pd
import pickle
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

df = pd.read_csv('normal.csv' , encoding='CP949')
df = fix_columns(df)

print(df)

df = df[['pitch_ATTITUDE', 'yaw_ATTITUDE', 'error_rp_AHRS',
       'error_yaw_AHRS', 'omegaIx_AHRS', 'omegaIy_AHRS', 'omegaIz_AHRS',
       'velocity_variance_EKF_STATUS_REPORT',
       'pos_horiz_variance_EKF_STATUS_REPORT',
       'pos_vert_variance_EKF_STATUS_REPORT',
       'compass_variance_EKF_STATUS_REPORT', 'nav_roll_NAV_CONTROLLER_OUTPUT',
       'nav_pitch_NAV_CONTROLLER_OUTPUT', 'xtrack_error_NAV_CONTROLLER_OUTPUT',
       'aspd_error_NAV_CONTROLLER_OUTPUT', 'xacc_RAW_IMU', 'yacc_RAW_IMU',
       'zacc_RAW_IMU', 'xgyro_RAW_IMU', 'ygyro_RAW_IMU', 'zgyro_RAW_IMU',
       'xmag_RAW_IMU', 'ymag_RAW_IMU', 'zmag_RAW_IMU',
       'servo1_raw_SERVO_OUTPUT_RAW', 'servo2_raw_SERVO_OUTPUT_RAW',
       'servo3_raw_SERVO_OUTPUT_RAW', 'servo4_raw_SERVO_OUTPUT_RAW',
       'servo5_raw_SERVO_OUTPUT_RAW', 'servo6_raw_SERVO_OUTPUT_RAW',
       'servo7_raw_SERVO_OUTPUT_RAW', 'servo8_raw_SERVO_OUTPUT_RAW',
       'vibration_x_VIBRATION', 'vibration_y_VIBRATION',
       'vibration_z_VIBRATION', 'roll_AHRS_DIFF', 'pitch_AHRS_DIFF',
       'yaw_AHRS_DIFF', 'xacc_IMU_DIFF', 'yacc_IMU_DIFF', 'zacc_IMU_DIFF',
       'xgyro_IMU_DIFF', 'ygyro_IMU_DIFF', 'zgyro_IMU_DIFF', 'xmag_IMU_DIFF',
       'ymag_IMU_DIFF', 'zmag_IMU_DIFF', 'roll_ATTITUDE']]
print(df.columns)
model = pickle.load(open('220119model_2', 'rb'))
print(df)

y_pred = model.predict(df)

import collections
print(collections.Counter(y_pred))
