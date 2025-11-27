import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# df = pd.read_csv("MNRAS_paper/csv_for_paper/df_test_for_paper.csv")

CSV_FILENAME = "df_valid.csv"

df = pd.read_csv(CSV_FILENAME)

# Point Spread Function : https://academic.oup.com/mnras/article/509/3/3966/6378289 1.18 for r

# petroRad_r_psf = petroRad_r * 2 * numpy.sqrt( 2 * numpy.log(2) ) / psfWidth_r

petroRad_r_unsized = np.array(df["petro_th90"] * 2 * np.sqrt(2*np.log(2)) / 1.18)

petroRad_r_unsized = petroRad_r_unsized.reshape(-1, 1)

scaler = MinMaxScaler()

# Fit to original paper scaling

scaler.fit([[0.07466520868457423], [589.4]])

# 589.4 is the maximum unsized value from original paper, used for scaling for continuity
# 0.07466520868457423 is the minimum value

scaler.data_max_ = 589.4

df["petroRad_r_psf"] = scaler.transform(petroRad_r_unsized)

df.to_csv(CSV_FILENAME.split('.', maxsplit=1)[0] + "_plus_petroRad_r.csv")
