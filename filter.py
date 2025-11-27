import pandas as pd

import os

df_train = pd.read_csv("df_train.csv")
df_valid = pd.read_csv("df_valid.csv")

print(df_valid.shape[0])

train_iaunames = [iauname for iauname in list(df_train["iauname"]) if os.path.exists("/scratch/christoq_root/christoq0/jjfisch/" + iauname)]
valid_iaunames = [iauname for iauname in list(df_valid["iauname"]) if os.path.exists("/scratch/christoq_root/christoq0/jjfisch/" + iauname)]

print(len(valid_iaunames))

df_train = df_train[df_train.iauname.isin(train_iaunames)]
df_valid = df_valid[df_valid.iauname.isin(valid_iaunames)]

print(df_valid.shape[0])

df_train.to_csv("df_train.csv")
df_valid.to_csv("df_valid.csv")

