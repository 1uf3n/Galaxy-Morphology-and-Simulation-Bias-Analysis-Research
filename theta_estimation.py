## IMPORTS
import pandas as pd
import numpy as np

from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

## LOAD DATA

# Load data and train, valid, test separation
path = 'csv_for_paper/'
# data = pd.read_csv(path +'data_for_nn.csv')
df_train = pd.read_csv(path +'df_train_for_paper.csv')
df_valid = pd.read_csv(path +'df_valid_for_paper.csv')
df_test = pd.read_csv(path +'df_test_for_paper.csv')

## FUNCTION TO GENERATE BINS OF ALPHA

def generate_bins(df, num, arg='num_per_bin', bin_mode=1, alpha_mode='mid'):
    # bin_mode:  0 -> Bins separation according to alfa
    #            1 -> Bins separation according to the number of instances


    # alpha_mode: 'mid'  -> takes the mid point of the interval
    #             'mean' -> takes the mean of the interval

    limits = []
    if bin_mode == 1:
        # According to number of instances
        alpha_tmp = df.copy()['petroRad_r_psf'].values
        alpha_tmp.sort()

        if arg == 'num_bins':
            num_bins = num
            num_per_bin = int(len(df.index)/num_bins) #original
            #print('Num per bin: ' + str(num_per_bin))

            i = 1
            limits.append(alpha_tmp[0])
            while i <= num_bins:
                if num_per_bin*i >= len(alpha_tmp):
                    limits.append(alpha_tmp[-1])
                    break
                else:
                    limits.append(alpha_tmp[num_per_bin*i])

                i += 1

        elif arg == 'num_per_bin':
            num_per_bin = num
            num_bins = int(np.ceil(len(df.index)/num_per_bin))

            i = 1
            limits.append(alpha_tmp[0])
            while i <= num_bins:
                if num_per_bin*i >= len(alpha_tmp):
                    limits.append(alpha_tmp[-1] + 1.0)
                    break
                else:
                    limits.append(alpha_tmp[num_per_bin*i])

                i += 1

    else:
        # According to alfa
        alpha = df[df['petroRad_r_psf'] <= 0.3]['petroRad_r_psf'].values
        lower_limit = np.min(alpha)
        upper_limit = np.max(alpha)
        bin_size = (upper_limit - lower_limit)/num_bins

        i = 1
        limits.append(lower_limit)
        while i <= num_bins:
            limits.append(limits[i-1] + bin_size)
            i = i+1


    limits[-1] += 1.0

    # Biased class fractions
    i = 0
    cf_ell = []
    cf_cs = []
    while i < num_bins:
        ell = df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1]) & (df['label']==0)].count()[0]
        cs = df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1]) & (df['label']==1)].count()[0]
        total = df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1])].count()[0]

        if total > 0:
            ell = ell/total
        if total > 0:
            cs = cs/total

        cf_ell.append(ell)
        cf_cs.append(cs)

        i += 1

    limits[-1] =- 1.0

    # Alphas
    alpha = []
    if alpha_mode == 'mid':
        for i in range(0, num_bins):
            alpha_tmp = ( limits[i] +  limits[i+1] )/2
            alpha.append(alpha_tmp)

    elif alpha_mode == 'mean':
        for i in range(0, num_bins):
            alpha_tmp = df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1])]['petroRad_r_psf']
            alpha_tmp = alpha_tmp.mean()
            alpha.append(alpha_tmp)
    elif alpha_mode == 'median':
        for i in range(0, num_bins):
            alpha_tmp = df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1])]['petroRad_r_psf']
            alpha_tmp = alpha_tmp.median()
            alpha.append(alpha_tmp)
    else:
        print('Input a valid alpha_mode')

    return np.array(cf_ell), np.array(cf_cs), np.array(alpha)


## FUNCTIONS TO OPTIMIZE

# p(y_bias = 0 | y_true = 1, alpha, theta)
def label_flipping_probability(alfa, theta):
    return np.exp( -1*( (alfa**2)/(2*theta**2) ) )

def loss_func_smooth_var_fractions(var):
    theta = var[0]
    fs = var[1]
    fd = var[2]

    sum = 0
    for i in range(0, len(cf_disk)):
        fs_estimated = fs + fd*label_flipping_probability(alpha[i], theta)
        sum += (cf_smooth[i] - fs_estimated)**2

    return sum

def loss_func_disk_var_fractions(var):
    theta = var[0]
    fs = var[1]
    fd = var[2]

    sum = 0
    for i in range(0, len(cf_disk)):
        fd_estimated = fd*(1 - label_flipping_probability(alpha[i], theta))
        sum += (cf_disk[i] - fd_estimated)**2

    return sum

# Constraints for optimization

# fs + fd == 1
def constraint1(vars_0):
    return vars_0[1] + vars_0[2] - 1

# theta >= 0.001
def constraint2(vars_0):
    return vars_0[0] - 0.001

# fs >= 0.00000001
def constraint3(vars_0):
    return vars_0[1] - 0.00000001

# fd >= 0.00000001
def constraint4(vars_0):
    return vars_0[2] - 0.00000001

# theta <= 1 (real value = 589)
def constraint5(vars_0):
    return -vars_0[0] + 1

con1 = {'type':'eq', 'fun': constraint1}
con2 = {'type':'ineq', 'fun': constraint2}
con3 = {'type':'ineq', 'fun': constraint3}
con4 = {'type':'ineq', 'fun': constraint4}
con5 = {'type':'ineq', 'fun': constraint5}

## ESTIMATE THETA

# Transform alpha values
scaler_alpha = MinMaxScaler().fit(data[['petroRad_r_psf']].values)

# Number of bins
num_bins = 1525

# Get bins
cf_smooth, cf_disk, alpha = generate_bins(df_train.copy(), num_bins, arg='num_bins', alpha_mode='mid')

# Initialize variables
theta_0 = 0.017
vars_0 = [theta_0, cf_smooth[-1], cf_disk[-1]]

# Optimization
res = minimize(loss_func_smooth_var_fractions, vars_0, method='SLSQP', options={'ftol': 1e-15, 'disp': False}, constraints=[con1, con2, con3, con4, con5])
theta = res.x[0]
fs = res.x[1]
fd = res.x[2]

# Inverse transfrom theta to original scale
theta = np.array(theta)
theta = scaler_alpha.inverse_transform(theta.reshape(-1, 1))

print(theta)