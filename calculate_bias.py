## IMPORTS AND SETUP

# Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Metrics and data
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Others
import sys
import time
import os

## LOAD DATA

# Load data and train, valid, test separation
path = 'code/csv_files/'
data = pd.read_csv(path +'data_for_nn.csv')
df_train = pd.read_csv(path +'df_train_for_paper.csv')
df_valid = pd.read_csv(path +'df_valid_for_paper.csv')
df_test = pd.read_csv(path +'df_test_for_paper.csv')

## DATA GENERATOR

class CustomDataGeneratorIMG(tf.keras.utils.Sequence):

    def __init__(self, dataframe, directory,
                x_col, y_col, batch_size,
                input_size=(424,424,3), shuffle=True):
    
        self.dataframe = dataframe.copy()
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.dataframe.index)

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __get_image(self, path, target_size):
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = tf.image.resize_with_crop_or_pad(image_arr, 207, 207)

        return image_arr/255.0

    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    def __get_data(self, batches):
        # Generate data containing batch_size samples

        filename_batch = batches[self.x_col['filename']]
        label_batch = batches[self.y_col]

        x0_batch = np.asarray([self.__get_image(self.directory+filename, self.input_size) for filename in filename_batch])
        y_batch = label_batch.values

        return x0_batch, y_batch

    def __getitem__(self, index):
        batches = self.dataframe[index * self.batch_size:(index + 1) * self.batch_size]
        x0, y = self.__get_data(batches)        
        return x0, y
    
    def __len__(self):
        return self.n // self.batch_size


## BIAS ESTIMATION METRICS

# 2014

# Bias estimation
def estimate_bias(df, NA_j, mode, mode1):

    limits = []
    # Bins separation according to alfa
    if mode1 == 0:
        lower_limit = df['petroRad_r_psf'].min()
        upper_limit = df['petroRad_r_psf'].max()
        bin_size = (upper_limit - lower_limit)/NA_j
    
        i = 1
        limits.append(lower_limit)
        while i <= NA_j:
            limits.append(limits[i-1] + bin_size)
            i = i+1
    
    # Bins separation according to the number of instances
    elif mode1 == 1:
        alfa_tmp = df.copy()['petroRad_r_psf'].values
        alfa_tmp.sort()
        num_per_bin = int(len(df.index)/NA_j)
        i = 1
        limits.append(alfa_tmp[0])
        while i <= NA_j:
            limits.append(alfa_tmp[num_per_bin*i - 1])
            i += 1

    #To include upper limit
    limits[-1] += 1.0

    r_el = -1
    r_cs = -1
    # r_k = fraction over the entire dataset
    if mode == 0:
        n_el = ((df['label'])==0).sum()
        n_cs = ((df['label'])==1).sum()
        n_total = n_el + n_cs

        r_el = n_el/n_total
        r_cs = n_cs/n_total

    # r_k = fraction in the least biased bin
    elif mode == 1:
        aux = len(limits) - 1
        n_el = df[(df['petroRad_r_psf']>=limits[aux-1]) & (df['petroRad_r_psf']<limits[aux]) & (df['label']==0)].count()[0]
        n_cs = df[(df['petroRad_r_psf']>=limits[aux-1]) & (df['petroRad_r_psf']<limits[aux]) & (df['label']==1)].count()[0]
        n_total = n_el + n_cs

        r_el = n_el/n_total
        r_cs = n_cs/n_total

    i = 0
    sigma_el = 0
    while i < NA_j:
        r_jkl = df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1]) & (df['label']==0)].count()[0]

        if r_jkl != 0:
            r_jkl = r_jkl/df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1])].count()[0]

        sigma_el += (r_jkl - r_el)**2
        i += 1

    sigma_el = sigma_el/NA_j

    i = 0
    sigma_cs = 0
    while i < NA_j:
        r_jkl = df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1]) & (df['label']==1)].count()[0]

        if r_jkl != 0:
            r_jkl = r_jkl/df[(df['petroRad_r_psf']>=limits[i]) & (df['petroRad_r_psf']<limits[i+1])].count()[0]
        
        sigma_cs += (r_jkl - r_cs)**2
        i += 1
    
    sigma_cs = sigma_cs/NA_j

    L = np.sqrt((sigma_el + sigma_cs)/2)

    return L


# 2018

#BiasAnalyser.py

#import pylab as pl
#from astropy.io import fits as pf
import numpy as np
import matplotlib as mpl
from scipy.interpolate import *
from matplotlib import cm, colors

class BiasAnalyser ():

    def __init__ (self, N = -1):
        self.N = N

    def getFractions (self, x, classes, labels, nx, ini_crit = True,
                      get_is = False, get_N = False, equalBins = False):

        # Create criteria for selecting sources that correspond to
        # desired labels.
        if np.isscalar (ini_crit):
            crit = np.where(np.in1d (classes, labels))
        else:
            crit = np.where(np.in1d (classes, labels))[ini_crit]

        x = x[crit]
        classes = classes[crit]

        N = len(x)

        # If no sources present return empy array.
        if N == 0:
            print("BiasAnalyzer.getFractions N = 0")
            ret = np.array([]), np.array([]), np.array([]), np.array([])
            if get_is:
                ret = ret + (np.array([]),)
            if get_N:
                ret += (N,)
            return ret

        # Create array containing equal number of sources per bin.
        ns = np.array(np.round(np.linspace (0, N, nx + 1)), dtype = int)
        i_s = x.argsort()
        x_means = []

        # Create array for fractions per bin.
        fractions = np.zeros((len(ns)-1, len(labels)))
        i_ret = []
        dx = (x[-1]-x[0]) / nx

        for i in range(len(ns) - 2):
            if equalBins:
                print(i, len(equalBins))
                x1 = equalBins[i]
                x2 = equalBins[i+1]
                i1 = i_s [(x[i_s] >= x1) & (x[i_s] < x2)]
                Ni = len(i1)
            else:
                # Select objects in bin i
                i1 = i_s [ns[i]:ns[i+1]]
                Ni = ns[i+1] - ns[i]
            x_means.append(x[i1].mean())

            # For each label calculate fractions
            for k in range(len(labels)):
                fractions[i, k] = 1.*(classes[i1] == labels[k]).sum()/Ni
            i_ret.append(i1)
        i1 = i_s [ns[-2]:]
        Ni = N - ns[-2]
        x_means.append(x[i1].mean())
        for k in range(len(labels)):
            fractions[len(ns) - 2, k] = 1.*(classes[i1] == labels[k]).sum()/Ni
        i_ret.append(i1)
        ret = np.array(x_means), fractions
        if get_is:
            ret = ret + (np.array(i_ret),)
        if get_N:
            ret += (N,)
        return ret

    # splits x into nx sorted vectors
    def getEqualNumberSplit (self, x, nx):
        N = len(x)
        dn = 1. * N / nx
        ns = np.array(np.round(np.linspace(0, N, nx+1)), dtype = "int")
        i_s = x.argsort()
        ret = []
        for i in range(len(ns) - 1):
            i1 = i_s [ns[i]:ns[i+1]]
            ret.append (i1)
        ret = np.array(ret)
        return ret

    def KDTree (self, x, pow_n, i_array, i_var = 0, i_ret = 0, 
                crit = "iterative", y = None, labels = None):
        if pow_n == 0:
            return np.zeros(len(i_array)) + i_ret
            
        if crit == "iterative":
            this_i_var = i_var
            sep = np.median(x[:, i_var])
            # print sep
        elif crit == "highest_fraction_difference":
            if y is None or labels is None:
                print("ERROR BiasAnalyser.KDTree: y and labels required for ", crit)
                exit()
            f_d = np.zeros(x.shape[1])  # Fraction differences
            seps = np.zeros(x.shape[1]) # Thresholds for each
                                        # intrinsic variable
            
            # calculate splitting fractions
            for i in range(x.shape[1]):
                seps[i] = np.median(x[:, i])
                if (x[:, i] < seps[i]).sum() == 0 or (x[:, i] >= seps[i]).sum() == 0:
                    seps[i] = np.mean(x[:, i])
                f_left = 0
                f_right = 0
                for k in labels:
                    if len(x) != len(y):
                        print(len(x[:, i] < seps[i]), len(y), len(x))
                        print((y[x[:, i] < seps[i]] == k).sum())
                        print((x[:, i] < seps[i]).sum())
                        exit()
                    f_left  = (1. * (y[x[:, i] < seps[i]] == k).sum() / 
                               (x[:, i] < seps[i]).sum())
                    f_right = (1. * (y[x[:, i] >= seps[i]] == k).sum() / 
                               (x[:, i] >= seps[i]).sum())
                    f_d[i] += np.abs (f_left - f_right)
                f_d [i] = f_d [i] / len (labels)
            this_i_var = np.argmax (f_d)
            sep = seps [this_i_var]
            # print " " , i_ret, " KDTree f_d = ", f_d, this_i_var, x.shape, f_d.prod()
            # if f_d.prod() == 0.:
            #     i = np.argmin (f_d)
            #     print seps[i]
            #     print i, (y[x[:, i] < seps[i]]).sum(), (y[x[:, i] < seps[i]]).sum()
            #     exit()

        else:
            print("ERROR BiasAnalyser.KDTree: ", crit, " criteria not implemented.")
            exit()

        crit_left = np.arange(len(x))[x[:, this_i_var] < sep]
        crit_right = np.arange(len(x))[x[:, this_i_var] >= sep]
        i_var_new = (this_i_var + 1)%x.shape[1]
        
        left = self.KDTree (x[crit_left], pow_n - 1, crit_left, 
                             i_var_new, i_ret, crit, y[crit_left], labels)
        right = self.KDTree (x[crit_right], pow_n - 1, crit_right, 
                             i_var_new, i_ret + 2**(pow_n-1), crit, y[crit_right], labels)
        ret = np.zeros(x.shape[0], dtype = "int")
        ret[crit_left] = left
        ret[crit_right] = right
        return ret
        
    def calculateSigma2 (self, intrinsic, observable, y, labels, 
                         log2_bins_int, bins_obs, increasing_bias, 
                         kd_tree = "iterative"):
        kd_tree = self.KDTree (intrinsic, log2_bins_int, 
                               np.arange(intrinsic.shape[0]), crit = kd_tree,
                               y = y, labels = labels)
        kd_keys = np.unique(kd_tree)
        sigma2 = np.zeros((observable.shape[1], len(labels), 2**log2_bins_int))
        
        for j in range (observable.shape[1]):
            for q in kd_keys:
                i_bin_int = (kd_tree == q)
                # rs.shape = (bins_obs, len(labels))
                fs, rs = self.getFractions (observable[:, j][i_bin_int], 
                                            y[i_bin_int], labels, bins_obs)
                for k in range (len(labels)):
                    if increasing_bias[j]:
                        sigma2 [j, k, q] = ((rs[:, k] - rs[0, k])**2).sum()/bins_obs
                    else:
                        sigma2 [j, k, q] = ((rs[:, k] - rs[-1, k])**2).sum()/bins_obs
        return sigma2
    
    def getFractionsPerObject (self, intrinsic, observable, y, labels, 
                               log2_bins_int, bins_obs, increasing_bias, 
                               kd_tree = "iterative"):
        kd_tree = self.KDTree (intrinsic, log2_bins_int, 
                               np.arange(intrinsic.shape[0]), crit = kd_tree,
                               y = y, labels = labels)
        kd_keys = np.unique(kd_tree)
        sigma2 = np.zeros((observable.shape[1], len(labels), 2**log2_bins_int))
        output_size = np.concatenate((observable.shape, [len(labels)]))
        #print("output_size = ", output_size)
        int_frac = np.zeros(output_size)
        obs_frac = np.zeros(output_size)

        for j in range (observable.shape[1]):
            for q in kd_keys:
                i_bin_int = (kd_tree == q)
                is_bin = np.arange(observable.shape[0])[i_bin_int]
                # rs.shape = (bins_obs, len(labels))
                fs, rs, i_s = self.getFractions (observable[:, j][i_bin_int], 
                                            y[i_bin_int], labels, bins_obs,
                                            get_is = True)
                for k in range (len(labels)):
                    for i in range(len(i_s)):
                        #print(is_bin[i_s[i]].min(), is_bin[i_s[i]].max())
                        obs_frac[is_bin[i_s[i]], j, k] = rs[i, k]
                        if increasing_bias[j]:
                            int_frac[is_bin[i_s[i]], j, k] = rs[0, k]
                            #sigma2 [j, k, q] = ((rs[:, k] - rs[0, k])**2).sum()/bins_obs
                        else:
                            int_frac[is_bin[i_s[i]], j, k] = rs[-1, k]
                            #sigma2 [j, k, q] = ((rs[:, k] - rs[-1, k])**2).sum()/bins_obs
        return int_frac, obs_frac
        

    def getLsBins (self, intrinsic, observable, y, labels, crit = True,
                   bins_in = (5,5), bins_ob = 20, equalN = True, 
                   minElementsBin = 100, increasing_bias = True):

        # create array of indices for the data
        if np.isscalar (crit):
            crit = np.arange(intrinsic.shape[0])

        Ls = np.zeros (bins_in)
        Ns = np.zeros (bins_in, dtype = int)

        # Check if there is enough data for minElementsBin objects per
        # bin
        N_data = intrinsic[crit].shape[0]
        N_objs_per_bin = bins_in.prod() * bins_ob * minElementsBin
        if (N_data < N_objs_per_bin):
            print("BiasAnalyser.getLsBins: not enough data", (Ls + 1).sum())
            print(" ", N_data, " < ", N_objs_per_bin)
            return Ls + 1, np.zeros (bins_in), np.zeros (bins_in), Ns

        # split tbdata into bins[0] vectors sorted in terms of field_r
        i_r = self.getEqualNumberSplit (intrinsic[:, 0][crit], bins_in[0])

        # equalN = True: same number of objs. per bin.
        if equalN:
            bins_r_mean = np.zeros (bins_in)
            bins_c_mean = np.zeros (bins_in)
        else:
            bins_r_mean = []
            bins_c_mean = []
            #calculate means on bins of field_r 
            for i in range(bins_in[0]):
                bins_r_mean.append(intrinsic[:, 0][crit][i_r[i]].mean())
    
        if not equalN:
            iBins = self.getEqualNumberSplit (intrinsic[:, 1], bins_in[1])
            fieldBins = []
            for i in range(bins[1]):
                fieldBins.append (intrinsic[:, 1][iBins[i]].min())
                bins_c_mean.append(intrinsic[:, 1][iBins[i]].mean())
            fieldBins.append(intrinsic[:, 1][iBins[-1]].max())
    
        for i in range(bins_in[0]):
            if equalN:
                # get bins in i_r[i] with equal number of objs in field_c
                i_c = self.getEqualNumberSplit (intrinsic[:, 1][crit][i_r[i]], 
                                                bins_in[1])
            else:
                i_c = []
                for j in range (bins[1]):
                    crit1 = ((intrinsic[:, 1][crit][i_r[i]] >= fieldBins[j]) & 
                             (intrinsic[:, 1][crit][i_r[i]] < fieldBins[j+1]))
                    i_c.append(np.arange(len(intrinsic[:, 1][crit][i_r[i]]))[crit1])
            for j in range(bins_in[1]):
                Ns[i][j] = len(observable[crit][i_r[i]][i_c[j]])

                # calculate mean field_r and field_c values
                if equalN:
                    bins_r_mean [i][j] = (intrinsic[:, 0][crit][i_r[i]][i_c[j]]).mean()
                    bins_c_mean [i][j] = (intrinsic[:, 1][crit][i_r[i]][i_c[j]]).mean()
                # if number of objects per bins < minElementsBin, L = -0.00001
                if len(observable[crit][i_r[i]][i_c[j]]) < minElementsBin / bins_ob:
                    Ls[i][j] = -0.00001
                    continue
                crit1 = np.arange(len(observable))[crit][i_r[i]][i_c[j]]

                # fs = means of field_plot per bin, rs = fractions of S
                fs, rs = self.getFractions (observable, y, labels, bins_ob,
                                            ini_crit = crit1)
                L_ij = 0
                for k in range(len(labels)):
                    if increasing_bias:
                        L_ij += np.sqrt(((rs[:, k] - rs[0, k])**2).sum()/bins_ob)
                    else:
                        L_ij += np.sqrt(((rs[:, k] - rs[-1, k])**2).sum()/bins_ob)
                Ls [i][j] = L_ij / len(labels)
        return Ls, bins_r_mean, bins_c_mean, Ns

    def L (self, intrinsic, observables, y, labels, increasing_bias, 
           log2_bins_int, bins_ob = 20, minElementsBin = 10, N = -1, 
           bootstrap = False, kd_tree = "iterative"):

        # define a criteria for choosing objects with labels in
        # "labels"
        crit = np.where(np.in1d (y, labels))[0]

        if N > len(crit):
            print("N > len(crit)", N, len(crit))
            return 0., 0.

        if N > 0 and N < len(crit):
            i_s = np.arange (len(crit))
            np.random.shuffle(i_s)
            crit = crit[i_s[:N]]     # N shuffled indices of class in
                                     # "labels"

        N = len(crit)
        if bootstrap:
            crit = np.random.choice (crit, size = N, replace = True)

        N_bins = 2**log2_bins_int*bins_ob
        if N < N_bins*minElementsBin:
            print("N < N_bins*minElementsBin", N, N_bins*minElementsBin)
            return 0., 0.

        Npl = observables.shape[1]
        L = 0
        sigma2 = self.calculateSigma2 (intrinsic[crit], observables[crit], y[crit], 
                                       labels, log2_bins_int, bins_ob, 
                                       increasing_bias, kd_tree = kd_tree)

        return np.sqrt(sigma2.sum()/ (np.prod(sigma2.shape))), N

    def getRandomL (self, intrinsic, observables, y, labels, increasing_bias,
                    N_calc, bins_in, bins_ob = 20, minElementsBin = 10, 
                    N_objs = -1, bootstrap = False, kd_tree = "iterative"):
        Ls = np.zeros(N_calc)
        Ns = np.zeros(N_calc)
        
        for i in range(N_calc):
            Ls[i], Ns[i] = self.L (intrinsic, observables, y, labels, 
                                   increasing_bias, bins_in, bins_ob, 
                                   minElementsBin, N_objs, bootstrap = bootstrap,
                                   kd_tree = kd_tree)
                                  # (intrinsic, observables, y, labels, 
                                  #  bins_in, bins_ob, minElementsBin, 
                                  #  N_objs, increasing_bias)
        return Ls, Ns


    def aux (self, intrinsic, observable, y, labels, 
             log2_bins_int, bins_obs, increasing_bias, 
             kd_tree = "iterative"):
        kd_tree = self.KDTree (intrinsic, log2_bins_int, 
                               np.arange(intrinsic.shape[0]), crit = kd_tree,
                               y = y, labels = labels)

        kd_keys = np.unique(kd_tree)
        sigma2 = np.zeros((observable.shape[1], len(labels), 2**log2_bins_int))
        fractions = []
        alfas = []
        
        for j in range (observable.shape[1]):
            for q in kd_keys:
                i_bin_int = (kd_tree == q)
                # rs.shape = (bins_obs, len(labels))
                fs, rs = self.getFractions (observable[:, j][i_bin_int], 
                                            y[i_bin_int], labels, bins_obs)
                
                fractions.append(rs)
                alfas.append(fs)
                
        return fractions, alfas, kd_tree


# Modified for pandas dataframe
def createLabels (tbdata, classf, pbb_thresholds):
    N_tot = len(tbdata)
    if not np.isscalar(pbb_thresholds):
        if not (pbb_thresholds is None) and len (classf) != len (pbb_thresholds):
            print("ERROR: the number of pbb fields does not match the number of thresholds.")
            exit()

        y = np.zeros(N_tot)
        if len(classf) == 1:
            y[tbdata['p_cs'] >= pbb_thresholds[0]] = 1
            y[1 - tbdata['p_cs'] > pbb_thresholds[0]] = 2
        else:
            for i in range(len(classf)):
                print(classf[i], classf, tbdata[classf[i]].shape)
                crit = (tbdata[classf[i]] >= pbb_thresholds[i])
                print(i, crit.sum())
                y[crit] = i + 1   

    else:
        if len(classf) != 1:
            #print "ERROR: classf can only be an array when using --pbb_thresholds."
            print("ERROR: Can use more than one 'classf' only when using --pbb_thresholds.")
            exit()
        y = tbdata[classf[0]]
    
    return y

def calculate_bias(dataframe, label_name, number_objects, int_pars, obs_pars, pbb_thresholds, 
                   no_zeros, bins_obs, log2_bins_int, N_iter, labels, verbose=False):


    N_objs = number_objects     # Number of objects for calculating bias.
    classf = label_name         # Label or probability column name.
    fields_int = int_pars       # Intrinsic parameters
    fields_beta = obs_pars      # Observable parameters
    N_calc = N_iter             # Number of calculations of L to calculate means and standard deviations.
    l2_bins_int = log2_bins_int # Number of bins for intrinsic pars.
    bins_obs = bins_obs         # Number of bins for observable pars.
    N_iter = N_iter             # Number of iterations for mean and std.
    minElementsBin = 5          # Min. number of objs. per bin.

    if pbb_thresholds:
        pbb_thresholds = np.array (pbb_thresholds).astype(np.float)
    else:
        pbb_thresholds = pbb_thresholds

    tbdata = dataframe
    N_tot = len(tbdata)

    #Randomize the data.
    tbdata = tbdata.sample(frac=1).reset_index(drop=True)

    if any(pbb_thresholds):
        y = createLabels (tbdata, classf, pbb_thresholds)
    else:
        y = np.array(tbdata[classf[0]], dtype = int)

    data_aux = tbdata.values
    if verbose:
        print(data_aux.shape)
    crit_zeros = np.ones (len(y), dtype = bool)

    if no_zeros:
        if verbose:
            print("NULL = ", pd.isnull(data_aux).any(axis = 1).sum())
        #crit_zeros = (y != 0) & ~np.isnan(data_aux).any(axis = 1)
        crit_zeros = (y != 0) & ~pd.isnull(data_aux).any(axis = 1)
    else:
        #crit_zeros = ~np.isnan(data_aux).any(axis = 1)
        crit_zeros = ~pd.isnull(data_aux).any(axis = 1)
    y = y[crit_zeros]

    # Get intrinsic parameters.
    intrinsic = []
    for intr in int_pars:
        intrinsic.append (tbdata[intr][crit_zeros])
    intrinsic = np.array(intrinsic).transpose()

    # Get observable parameters.
    observables = []
    for obs in fields_beta:
        observables.append (tbdata[obs][crit_zeros])
    observables = np.array(observables).transpose()

    if labels:
        labels = np.array(labels, dtype = int)
    else:
        labels = np.unique(y)

    if verbose:    
        print("labels = ", labels)
    tot_lab = 0
    for i in range (len(labels)):
        if verbose:
            print("class ", labels[i], " = ", (y == labels[i]).sum())
        tot_lab += (y == labels[i]).sum()
    
    if verbose:
        print("Labeled total = ", tot_lab)

    # Ls, Ns = dataAn.getRandomL (intrinsic, observables, y, labels, N_iter, bins_int,
    #                             bins_obs, minElementsBin, N_objs, 
    #                             [True, True, False])
    increasing_bias = [True, False]
    if verbose:
        print(N_objs, l2_bins_int, bins_obs)
        print(intrinsic.shape, observables.shape, y.shape, labels, increasing_bias, N_calc, l2_bins_int, bins_obs, N_objs, l2_bins_int, bins_obs)

    dataAn = BiasAnalyser()
    Ls, Ns = dataAn.getRandomL (intrinsic, observables, y, labels, 
                                increasing_bias, N_calc, l2_bins_int,
                                bins_obs, minElementsBin = N_objs, 
                                N_objs = -1,
                                bootstrap = False, 
                                kd_tree = "iterative")
    #                            kd_tree = "highest_fraction_difference")

    if verbose:
        print("-------")
        
    print("L = ", Ls.mean(), " +- ", Ls.std())
    print("$", np.round(Ls.mean(), 3), " \pm ", np.round(Ls.std(), 3), "$")



## MODEL

base_model = ResNet50(include_top=False,
                      weights=None,
                      input_shape=(207, 207, 3),
                      pooling=None)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

resnet_50 = keras.Model(inputs=base_model.input, outputs=output)


## PREDICT

batch_size = 32
input_shape = (207, 207, 3)

df_test['filename'] = df_test['filename'].apply(
    lambda f: os.path.join(f[:4], f)  # puts "J000/filename.png"
)

test_generator_debiasing = CustomDataGeneratorIMG(
    dataframe=df_test,
    directory='Galaxy_research/image/',
    x_col={'filename':'filename'},
    y_col='label',
    batch_size=batch_size,
    input_size=input_shape,
    shuffle=False
)

# Load weights
weights_path = "code/weights/best.weights.h5"
resnet_50.load_weights(weights_path)
print(f"\nWeights loaded ({weights_path})\n")

probabilities = resnet_50.predict(test_generator_debiasing, verbose=1)

predictions = np.argmax(probabilities, axis=1)

cnn_debiased = df_test.copy()
cnn_debiased['label'] = predictions
cnn_debiased['p_smooth_debiased'] = probabilities[:, 0]
cnn_debiased['p_disk_debiased'] = probabilities[:, 1]

## ESTIMATE BIAS

df = df_test.copy()
df_debiased = cnn_debiased.copy()
df_debiased['org_label'] = df['label'].values

# mode: 0 -> r_k = fraction over the entire dataset
#       1 -> r_k = fraction in the least biased bin
mode = 1

# mode1: 0 -> Bins separation according to alfa
#        1 -> Bins separation according to the number of instances
mode1 = 1
NA_j = 10

L_original = estimate_bias(df.copy()[['petroRad_r_psf', 'label']], NA_j, mode, mode1)
L_debiased = estimate_bias(df_debiased.copy()[['petroRad_r_psf', 'label']], NA_j, mode, mode1)

print('C2014\n')
print(L_original, L_debiased)
print()

# Args
label_name = ['label']
int_pars = ["petroRad_r_kpc","absPetroMag_r", "z_gz"] # Name of the columns used as intrinsic parameters.
obs_pars = ["petroRad_r_psf"] # Name of the columns used as observable parameters.
pbb_thresholds = []
no_zeros = False # Do not consider labels that don't match the pbb. thresholds if True
N_iter = 1
labels = [0, 1] # Labels to be used.


bins_obs = 8
log2_bins_int = 3
number_objects = 190

verbose = False
print('C2018\n')
print('L original')
calculate_bias(df, label_name, number_objects, int_pars, obs_pars, pbb_thresholds,
            no_zeros, bins_obs, log2_bins_int, N_iter, labels, verbose=verbose)
print()

print('L debiased')
calculate_bias(df_debiased, label_name, number_objects, int_pars, obs_pars, pbb_thresholds,
            no_zeros, bins_obs, log2_bins_int, N_iter, labels, verbose=verbose)
print()

# Difference
difference = df_debiased[df_debiased['label'] != df_debiased['org_label']]
ell_to_sp = df_debiased[(df_debiased['label'] == 1) & (df_debiased['org_label'] == 0)]
sp_to_ell = df_debiased[(df_debiased['label'] == 0) & (df_debiased['org_label'] == 1)]

print('Difference: ' + str(len(difference)))
print('Elliptical to spiral: ' + str(len(ell_to_sp)))
print('Spiral to elliptical: ' + str(len(sp_to_ell)))
print(len(difference), len(ell_to_sp), len(sp_to_ell))

if len(sys.argv) > 2:
	difference.to_csv(sys.argv[2] + 'difference_all_data.csv', index=False)
	print()
	print('Difference (all data) saved to ' + sys.argv[2] + 'difference_all_data.csv')

	ell_to_sp.to_csv(sys.argv[2] + 'diff_ell_to_sp.csv', index=False)
	print('Difference (elliptical to spiral) saved to ' + sys.argv[2] + 'diff_ell_to_sp.csv')

	ell_to_sp_ra_dec = ell_to_sp[['ra', 'dec']]
	ell_to_sp_ra_dec.to_csv(sys.argv[2] + 'diff_ell_to_sp_ra_dec.csv', index=False)
	print('Difference (elliptical to spiral, only ra,dec) saved to ' + sys.argv[2] + 'diff_ell_to_sp_ra_dec.csv')

	sp_to_ell.to_csv(sys.argv[2] + 'diff_sp_to_ell.csv', index=False)
	print('Difference (spiral to elliptical) saved to ' + sys.argv[2] + 'diff_sp_to_ell.csv')

	sp_to_ell_ra_dec = sp_to_ell[['ra', 'dec']]
	sp_to_ell_ra_dec.to_csv(sys.argv[2] + 'diff_sp_to_ell_ra_dec.csv', index=False)
	print('Difference (spiral to elliptical, only (ra,dec)) saved to ' + sys.argv[2] + 'diff_sp_to_ell_ra_dec.csv')


print()
# Equals
equals = df_debiased[df_debiased['label'] == df_debiased['org_label']]
ell = equals[(equals['label'] == 0)]
sp = equals[(equals['label'] == 1)]

print('Equals: ' + str(len(equals)))
print('Elliptical: ' + str(len(ell)))
print('Spiral: ' + str(len(sp)))
print(len(equals), len(ell), len(sp))

if len(sys.argv) > 2:
	equals.to_csv(sys.argv[2] + 'equals_all_data.csv', index=False)
	print()
	print('Equals (all data) saved to ' + sys.argv[2] + 'equals_all_data.csv')

	ell.to_csv(sys.argv[2] + 'equals_ell.csv', index=False)
	print('Equals (only elliptical) saved to ' + sys.argv[2] + 'equals_ell.csv')

	ell_ra_dec = ell[['ra', 'dec']]
	ell_ra_dec.to_csv(sys.argv[2] + 'equals_ell_ra_dec.csv', index=False)
	print('Equals (only elliptical, only ra,dec) saved to ' + sys.argv[2] + 'equals_ell_ra_dec.csv')

	sp.to_csv(sys.argv[2] + 'equals_sp.csv', index=False)
	print('Equals (only spiral) saved to ' + sys.argv[2] + 'equals_sp.csv')

	sp_ra_dec = sp[['ra', 'dec']]
	sp_ra_dec.to_csv(sys.argv[2] + 'equals_sp_ra_dec.csv', index=False)
	print('Equals (only spiral, only (ra,dec)) saved to ' + sys.argv[2] + 'equals_sp_ra_dec.csv')

	
	