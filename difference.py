import pandas as pd
import os
import sys

dirlist = os.listdir(r"D:\SMT-AI-VOUCHER\220118\\")  # returns list
dirlist.remove('DIFF')
print(len(dirlist))

if __name__ == "__main__":

    # for dlist in dirlist:
    #     print(dlist)
    #     df = pd.read_csv(r"D:\SMT-AI-VOUCHER\220118\\" + dlist)
    #     df["DIFF"] = df.iloc[:, -1] - df.iloc[:, -2]
    #     df.to_csv(r"D:\SMT-AI-VOUCHER\220118\\DIFF\\" + dlist + "_DIFF.csv")

    newdf = pd.DataFrame()
    dirlist = os.listdir(r"D:\SMT-AI-VOUCHER\220118\\DIFF")

    for dlist in dirlist:
        df = pd.read_csv(r"D:\SMT-AI-VOUCHER\220118\\DIFF\\" + dlist)
        newdf[dlist] = df.DIFF.describe()
        # print(newdf)

    newdf.to_csv('4model_diff_describe.csv' , encoding='CP949' , index =False)
