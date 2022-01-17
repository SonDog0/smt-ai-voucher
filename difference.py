import pandas as pd
import os
import sys

dirlist = os.listdir(r"D:\220116_cross_test\RIDGE\cross_test")  # returns list

if __name__ == "__main__":

    for dlist in dirlist:
        df = pd.read_csv(r"D:\220116_cross_test\RIDGE\cross_test\\" + dlist)
        df["DIFF"] = df.iloc[:, -1] - df.iloc[:, -2]
        df.to_csv(r"D:\220116_cross_test\RIDGE\DIFFERENCE\\" + dlist + "_DIFF.csv")
