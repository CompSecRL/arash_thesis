#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

TEST_MODE = 0 # Testing macro

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

if TEST_MODE:
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# !pip install --upgrade pip
# !pip install python-docx
# !pip install antropy

MAGENTA = (202/255, 18/255, 125/255)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dataclasses
import math as math
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_validate, RandomizedSearchCV
import statsmodels.stats.api as sms
from tqdm.auto import tqdm
from dataclasses import asdict
from sklearn import svm
from tqdm import tqdm
import warnings
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, accuracy_score, make_scorer, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold # Feature selector
from sklearn.model_selection import KFold

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import iqr
from scipy.stats import median_abs_deviation
from scipy.stats import mode
from scipy.signal import find_peaks
from scipy.signal import peak_widths
# from scipy.special import entr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler, PowerTransformer
get_new_scaler_dict = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "Normalizer": Normalizer, 
                       "MaxAbsScaler": MaxAbsScaler, "RobustScaler": RobustScaler, "PowerTransformer": PowerTransformer}
from sklearn.preprocessing import normalize
from sklearn.metrics import auc
# import antropy as ant
import time
# import docx
from matplotlib.ticker import MaxNLocator


# Global utitlity functions are in separate notebook
print("utility_functions imports setup complete")


# In[3]:


def matchAccelGyroData(accel, gyro):
    # Match the numbers by merge_asof to the higher length vector
    accel_count = accel.count().time_stamp
    gyro_count = gyro.count().time_stamp
    names =['sensor_id', 'time_stamp', 'x', 'y', 'z']
    if accel_count > gyro_count:
        df = pd.merge_asof(accel, gyro, on="time_stamp", direction='nearest')
        df = df.sort_values(by=['time_stamp'])
        df = df.dropna()
        accel = df[["sensor_id_x", "time_stamp", "x_x", "y_x", "z_x"]]
        gyro = df[["sensor_id_y", "time_stamp", "x_y", "y_y", "z_y"]]
    else:
        df = pd.merge_asof(gyro, accel, on="time_stamp", direction='nearest')
        df = df.sort_values(by=['time_stamp'])
        df = df.dropna()
        gyro = df[["sensor_id_x", "time_stamp", "x_x", "y_x", "z_x"]]
        accel = df[["sensor_id_y", "time_stamp", "x_y", "y_y", "z_y"]]

    accel.columns = names
    gyro.columns = names
    
    return {'accel': accel, 'gyro': gyro}


# In[21]:


def getDataStats1(i, print_accel_gyro_array_size=1, print_na_df_array_size=0, begin_idx=500, end_idx=-500):
    
    #load the data
    names =['sensor_id', 'time_stamp', 'x', 'y', 'z']
#     if i!=8:
#         data = pd.read_csv('../input/wearable-assisted-ca/user{}_1.csv'.format(i), error_bad_lines = False, header=None, usecols = range(len(names)))
#     else:
    
    data = pd.read_csv(f'{os.getcwd()}/WACA_dataset/user{i}_1.csv', error_bad_lines = False, header=None, usecols = range(len(names)), dtype = str)
    data.columns = names
    data = data[(data.sensor_id == '10') | (data.sensor_id =='4')]
    data.head(10)


    types_dict = {'sensor_id': 'int32', 'time_stamp': 'float64', 'x': 'float64', 'y': 'float64', 'z': 'float64'}
    for col, col_type in types_dict.items():
        data[col] = data[col].astype(col_type)

    # find how many NAN values in the data
    data.isna().sum()

    # since only 7 NAN is a very small amount, drop them
    df = data.dropna()

    # get rid of begin and end noise
    #sort df and filter
    df = df.sort_values(by=['time_stamp'])
    df = df[begin_idx:end_idx]
    
    # cleaning extreme outliers
    df = df[(df.x < 10.1) & ( -10.1 < df.x) & (df.y < 10.1) & ( -10.1 < df.y) & (df.z < 10.1) & ( -10.1 < df.z) ]

    # Extract Accelerometer values and sort
    accel = df[df.sensor_id == 10]
    accel = accel.sort_values(by=['time_stamp'])

    # Extract gyro values and sort
    gyro = df[df.sensor_id == 4]
    gyro = gyro.sort_values(by=['time_stamp'])
    
    if print_accel_gyro_array_size:
        print("{}) accel_count: {}, gyro_count: {}".format(i, accel.count().time_stamp, gyro.count().time_stamp))
    
    result = matchAccelGyroData(accel, gyro)
    accel, gyro = result['accel'], result['gyro']
    
    accel['EMA_x_a'] = accel['x'].ewm(span=40,adjust=False).mean()
    accel['EMA_y_a'] = accel['y'].ewm(span=40,adjust=False).mean()
    accel['EMA_z_a'] = accel['z'].ewm(span=40,adjust=False).mean()

    gyro['EMA_x_g'] = gyro['x'].ewm(span=40,adjust=False).mean()
    gyro['EMA_y_g'] = gyro['y'].ewm(span=40,adjust=False).mean()
    gyro['EMA_z_g'] = gyro['z'].ewm(span=40,adjust=False).mean()
    
    accel['EMA_x_a'] = accel['x']
    accel['EMA_y_a'] = accel['y']
    accel['EMA_z_a'] = accel['z']

    gyro['EMA_x_g'] = gyro['x']
    gyro['EMA_y_g'] = gyro['y']
    gyro['EMA_z_g'] = gyro['z']
    
    left = accel[["time_stamp", "EMA_x_a", "EMA_y_a", "EMA_z_a"]]
    right = gyro[["time_stamp", "EMA_x_g", "EMA_y_g", "EMA_z_g"]].set_index('time_stamp')
    df = left.join(right, on='time_stamp')

    if print_na_df_array_size:
        print("{}) na_count: {}, df count: {}".format(i, df.isna().sum().sum(), df.count().time_stamp))
    
    return {"accel":accel.count().time_stamp, "gyro": gyro.count().time_stamp, "df": df, "userIdx": i}


# In[29]:


def getDataStats2(i, print_accel_gyro_array_size=1, print_na_df_array_size=0, begin_idx=500, end_idx=-500):
    
    #load the data
    names =['sensor_id', 'time_stamp', 'x', 'y', 'z']
#     if i!=8:
#         data = pd.read_csv('../input/wearable-assisted-ca/user{}_1.csv'.format(i), error_bad_lines = False, header=None, usecols = range(len(names)))
#     else:
    
    data = pd.read_csv(f'{os.getcwd()}/WACA_dataset/user{i}_2.csv', error_bad_lines = False, header=None, usecols = range(len(names)), dtype = str)
    data.columns = names
    data = data[(data.sensor_id == '10') | (data.sensor_id =='4')]
    data.head(10)


    types_dict = {'sensor_id': 'int32', 'time_stamp': 'float64', 'x': 'float64', 'y': 'float64', 'z': 'float64'}
    for col, col_type in types_dict.items():
        data[col] = data[col].astype(col_type)

    # find how many NAN values in the data
    data.isna().sum()

    # since only 7 NAN is a very small amount, drop them
    df = data.dropna()

    # get rid of begin and end noise
    #sort df and filter
    df = df.sort_values(by=['time_stamp'])
    df = df[begin_idx:end_idx]
    
    # cleaning extreme outliers
    df = df[(df.x < 10.1) & ( -10.1 < df.x) & (df.y < 10.1) & ( -10.1 < df.y) & (df.z < 10.1) & ( -10.1 < df.z) ]

    # Extract Accelerometer values and sort
    accel = df[df.sensor_id == 10]
    accel = accel.sort_values(by=['time_stamp'])

    # Extract gyro values and sort
    gyro = df[df.sensor_id == 4]
    gyro = gyro.sort_values(by=['time_stamp'])
    
    if print_accel_gyro_array_size:
        print("{}) accel_count: {}, gyro_count: {}".format(i, accel.count().time_stamp, gyro.count().time_stamp))
    
    result = matchAccelGyroData(accel, gyro)
    accel, gyro = result['accel'], result['gyro']
    
#     accel['EMA_x_a'] = accel['x'].ewm(span=40,adjust=False).mean()
#     accel['EMA_y_a'] = accel['y'].ewm(span=40,adjust=False).mean()
#     accel['EMA_z_a'] = accel['z'].ewm(span=40,adjust=False).mean()

#     gyro['EMA_x_g'] = gyro['x'].ewm(span=40,adjust=False).mean()
#     gyro['EMA_y_g'] = gyro['y'].ewm(span=40,adjust=False).mean()
#     gyro['EMA_z_g'] = gyro['z'].ewm(span=40,adjust=False).mean()

    accel['EMA_x_a'] = accel['x']
    accel['EMA_y_a'] = accel['y']
    accel['EMA_z_a'] = accel['z']

    gyro['EMA_x_g'] = gyro['x']
    gyro['EMA_y_g'] = gyro['y']
    gyro['EMA_z_g'] = gyro['z']
    
    left = accel[["time_stamp", "EMA_x_a", "EMA_y_a", "EMA_z_a"]]
    right = gyro[["time_stamp", "EMA_x_g", "EMA_y_g", "EMA_z_g"]].set_index('time_stamp')
    df = left.join(right, on='time_stamp')

    if print_na_df_array_size:
        print("{}) na_count: {}, df count: {}".format(i, df.isna().sum().sum(), df.count().time_stamp))
    
    return {"accel":accel.count().time_stamp, "gyro": gyro.count().time_stamp, "df": df, "userIdx": i}


# In[30]:


# sample_rate = 10 #Hz
# #3352843.3
# x = np.array([318.45,302.78,316.47,334.14,333.41,326.15,320.07,318.68,314.12,308.64,
#               300.15,304.33,318.42,322.72,329.56,339.18,338.03,343.27,351.44,353.23,
#               352.35,352.88,353.43,352.14,351.28,352.82,353.36,353.35,353.19,353.82])

# mn=np.mean(x)
# print(f' mean = {mn:.3f} unit')
# print(f' sum x[i]**2  : {np.sum(x**2) :.1f} unit^2 ')


# print(f' n *sum X[k]**2   : {spectral_energy(x) :.1f} unit^2 ')


# In[ ]:


def spectral_energy(x):
    '''
    spectral_energy according to Parseval's theorem
    '''
    return (1/len(x)) * np.sum(np.abs(np.fft.rfft(x))**2)


# In[ ]:


def signal_to_encoding(signal_df):
    dic = {}
    
#     print("mean calculation started")
    dic['mean_x_a'] = np.mean(signal_df['EMA_x_a'])
    dic['mean_y_a'] = np.mean(signal_df['EMA_y_a'])
    dic['mean_z_a'] = np.mean(signal_df['EMA_z_a'])
    dic['mean_x_g'] = np.mean(signal_df['EMA_x_g'])
    dic['mean_y_g'] = np.mean(signal_df['EMA_y_g'])
    dic['mean_z_g'] = np.mean(signal_df['EMA_z_g'])
#     print("mean calculation ended")
    
#     print("median calculation started")
    dic['median_x_a'] = np.median(signal_df['EMA_x_a'])
    dic['median_y_a'] = np.median(signal_df['EMA_y_a'])
    dic['median_z_a'] = np.median(signal_df['EMA_z_a'])
    dic['median_x_g'] = np.median(signal_df['EMA_x_g'])
    dic['median_y_g'] = np.median(signal_df['EMA_y_g'])
    dic['median_z_g'] = np.median(signal_df['EMA_z_g'])
#     print("median calculation ended")
    
#     print("var calculation started")
    dic['var_x_a'] = np.var(signal_df['EMA_x_a'])
    dic['var_y_a'] = np.var(signal_df['EMA_y_a'])
    dic['var_z_a'] = np.var(signal_df['EMA_z_a'])
    dic['var_x_g'] = np.var(signal_df['EMA_x_g'])
    dic['var_y_g'] = np.var(signal_df['EMA_y_g'])
    dic['var_z_g'] = np.var(signal_df['EMA_z_g'])
#     print("var calculation ended")
    
#     print("avg absolute difference of peaks calculation started")
    peaks_x_a, _ = find_peaks(signal_df['EMA_x_a'])
    peaks_y_a, _ = find_peaks(signal_df['EMA_y_a'])
    peaks_z_a, _ = find_peaks(signal_df['EMA_z_a'])
    peaks_x_g, _ = find_peaks(signal_df['EMA_x_g'])
    peaks_y_g, _ = find_peaks(signal_df['EMA_y_g'])
    peaks_z_g, _ = find_peaks(signal_df['EMA_z_g'])
    
#     print(type(peak_widths(peaks_x_a, signal_df['EMA_x_a'], rel_height=0.5)[0]))
    dic['aadp_x_a'] = np.mean(peak_widths(signal_df['EMA_x_a'], peaks_x_a, rel_height=0.5)[0])
    dic['aadp_y_a'] = np.mean(peak_widths(signal_df['EMA_y_a'], peaks_y_a, rel_height=0.5)[0])
    dic['aadp_z_a'] = np.mean(peak_widths(signal_df['EMA_z_a'], peaks_z_a, rel_height=0.5)[0])
    dic['aadp_x_g'] = np.mean(peak_widths(signal_df['EMA_x_g'], peaks_x_g, rel_height=0.5)[0])
    dic['aadp_y_g'] = np.mean(peak_widths(signal_df['EMA_y_g'], peaks_y_g, rel_height=0.5)[0])
    dic['aadp_z_g'] = np.mean(peak_widths(signal_df['EMA_z_g'], peaks_z_g, rel_height=0.5)[0])
#     print("avg absolute difference of peaks calculation ended")
    
#     print("range calculation started")
    dic['ptp_x_a'] = np.ptp(signal_df['EMA_x_a'])
    dic['ptp_y_a'] = np.ptp(signal_df['EMA_y_a'])
    dic['ptp_z_a'] = np.ptp(signal_df['EMA_z_a'])
    dic['ptp_x_g'] = np.ptp(signal_df['EMA_x_g'])
    dic['ptp_y_g'] = np.ptp(signal_df['EMA_y_g'])
    dic['ptp_z_g'] = np.ptp(signal_df['EMA_z_g'])
#     print("range calculation ended")
    
#     print("mode calculation started")
    dic['mode_x_a'] = mode(signal_df['EMA_x_a'])[0][0]
    dic['mode_y_a'] = mode(signal_df['EMA_y_a'])[0][0]
    dic['mode_z_a'] = mode(signal_df['EMA_z_a'])[0][0]
    dic['mode_x_g'] = mode(signal_df['EMA_x_g'])[0][0]
    dic['mode_y_g'] = mode(signal_df['EMA_y_g'])[0][0]
    dic['mode_z_g'] = mode(signal_df['EMA_z_g'])[0][0]
#     print("mode calculation ended")
    
#     print("cov calculation started")
    dic['cov_x_a'] = np.cov(signal_df['EMA_x_a']) * 1
    dic['cov_y_a'] = np.cov(signal_df['EMA_y_a']) * 1
    dic['cov_z_a'] = np.cov(signal_df['EMA_z_a']) * 1
    dic['cov_x_g'] = np.cov(signal_df['EMA_x_g']) * 1
    dic['cov_y_g'] = np.cov(signal_df['EMA_y_g']) * 1
    dic['cov_z_g'] = np.cov(signal_df['EMA_z_g']) * 1
#     print("cov calculation ended")
    
#     print("mean absolute deviation calculation started")
    dic['mad_x_a'] = median_abs_deviation(signal_df['EMA_x_a'])
    dic['mad_y_a'] = median_abs_deviation(signal_df['EMA_y_a'])
    dic['mad_z_a'] = median_abs_deviation(signal_df['EMA_z_a'])
    dic['mad_x_g'] = median_abs_deviation(signal_df['EMA_x_g'])
    dic['mad_y_g'] = median_abs_deviation(signal_df['EMA_y_g'])
    dic['mad_z_g'] = median_abs_deviation(signal_df['EMA_z_g'])
#     print("mean absolute deviation calculation ended")
    
#     print("inter-quartile range calculation started")
    dic['iqr_x_a'] = iqr(signal_df['EMA_x_a'])
    dic['iqr_y_a'] = iqr(signal_df['EMA_y_a'])
    dic['iqr_z_a'] = iqr(signal_df['EMA_z_a'])
    dic['iqr_x_g'] = iqr(signal_df['EMA_x_g'])
    dic['iqr_y_g'] = iqr(signal_df['EMA_y_g'])
    dic['iqr_z_g'] = iqr(signal_df['EMA_z_g'])
#     print("inter-quartile range calculation ended")
    
#     print("correlation calculation started")
    dic['correlate_xy_a'] = np.correlate(signal_df['EMA_x_a'], signal_df['EMA_y_a'])[0]
    dic['correlate_yz_a'] = np.correlate(signal_df['EMA_y_a'], signal_df['EMA_z_a'])[0]
    dic['correlate_xz_a'] = np.correlate(signal_df['EMA_x_a'], signal_df['EMA_z_a'])[0]
    dic['correlate_xy_g'] = np.correlate(signal_df['EMA_x_g'], signal_df['EMA_y_g'])[0]
    dic['correlate_yz_g'] = np.correlate(signal_df['EMA_y_g'], signal_df['EMA_z_g'])[0]
    dic['correlate_xz_g'] = np.correlate(signal_df['EMA_x_g'], signal_df['EMA_z_g'])[0]
#     print("correlation calculation ended")
    
#     print("skew calculation started")
    dic['skew_x_a'] = skew(signal_df['EMA_x_a'])
    dic['skew_y_a'] = skew(signal_df['EMA_y_a'])
    dic['skew_z_a'] = skew(signal_df['EMA_z_a'])
    dic['skew_x_g'] = skew(signal_df['EMA_x_g'])
    dic['skew_y_g'] = skew(signal_df['EMA_y_g'])
    dic['skew_z_g'] = skew(signal_df['EMA_z_g'])
#     print("skew calculation ended")
    
#     print("kurtosis calculation started")
    dic['kurtosis_x_a'] = kurtosis(signal_df['EMA_x_a'])
    dic['kurtosis_y_a'] = kurtosis(signal_df['EMA_y_a'])
    dic['kurtosis_z_a'] = kurtosis(signal_df['EMA_z_a'])
    dic['kurtosis_x_g'] = kurtosis(signal_df['EMA_x_g'])
    dic['kurtosis_y_g'] = kurtosis(signal_df['EMA_y_g'])
    dic['kurtosis_z_g'] = kurtosis(signal_df['EMA_z_g'])
#     print("kurtosis calculation ended")
    
    
#     print("spectral energy calculation started")
    dic['spectral_energy_x_a'] = spectral_energy(signal_df['EMA_x_a'])
    dic['spectral_energy_y_a'] = spectral_energy(signal_df['EMA_y_a'])
    dic['spectral_energy_z_a'] = spectral_energy(signal_df['EMA_z_a'])
    dic['spectral_energy_x_g'] = spectral_energy(signal_df['EMA_x_g'])
    dic['spectral_energy_y_g'] = spectral_energy(signal_df['EMA_y_g'])
    dic['spectral_energy_z_g'] = spectral_energy(signal_df['EMA_z_g'])
#     print("spectral energy calculation ended")


#     print("spectral entropy calculation started")
#     method = 'fft'
#     normalize = False
# #     print(signal_df['EMA_x_a'])
# #     print(signal_df['EMA_x_a'].shape)
#     axis = -1
#     dic['spectral_entropy_x_a'] = ant.spectral_entropy(signal_df['EMA_x_a'], sf=len(signal_df['EMA_x_a']), method=method, normalize=normalize, axis=axis)
#     dic['spectral_entropy_y_a'] = ant.spectral_entropy(signal_df['EMA_y_a'], sf=len(signal_df['EMA_y_a']), method=method, normalize=normalize, axis=axis)
#     dic['spectral_entropy_z_a'] = ant.spectral_entropy(signal_df['EMA_z_a'], sf=len(signal_df['EMA_z_a']), method=method, normalize=normalize, axis=axis)
#     dic['spectral_entropy_x_g'] = ant.spectral_entropy(signal_df['EMA_x_g'], sf=len(signal_df['EMA_x_g']), method=method, normalize=normalize, axis=axis)
#     dic['spectral_entropy_y_g'] = ant.spectral_entropy(signal_df['EMA_y_g'], sf=len(signal_df['EMA_y_g']), method=method, normalize=normalize, axis=axis)
#     dic['spectral_entropy_z_g'] = ant.spectral_entropy(signal_df['EMA_z_g'], sf=len(signal_df['EMA_z_g']), method=method, normalize=normalize, axis=axis)
    
#     print(dic['spectral_entropy_x_a'],
#               dic['spectral_entropy_y_a'],
#               dic['spectral_entropy_z_a'],
#               dic['spectral_entropy_x_g'],
#               dic['spectral_entropy_y_g'],
#               dic['spectral_entropy_z_g'])
#     print("spectral entropy calculation ended")


#     print("entropy calculation started")
    
#     cols = signal_df[["EMA_x_a", "EMA_y_a", "EMA_z_a", "EMA_x_g", "EMA_y_g", "EMA_z_g"]]
#     cols = normalize(cols, norm='l2', axis = 0)
#     print(cols.sum(axis = 0))
#     cols = StandardScaler().fit_transform(cols)
#     p = cols/cols.sum(axis=0)
#     print(p.sum(axis=0))
#     print(p.shape)
#     print(cols.sum(axis=0))
#     entropy = entr(p).sum(axis=0)
#     print(entropy.shape)
#     dic['entropy_x_a'] = entropy[0]
#     dic['entropy_y_a'] = entropy[1]
#     dic['entropy_z_a'] = entropy[2]
#     dic['entropy_x_g'] = entropy[3]
#     dic['entropy_y_g'] = entropy[4]
#     dic['entropy_z_g'] = entropy[5]
#     print("entropy calculation ended")
    
    vector = [dic['mean_x_a'], 
              dic['mean_y_a'],
              dic['mean_z_a'],
              dic['mean_x_g'],
              dic['mean_y_g'],
              dic['mean_z_g'],
              
              dic['median_x_a'],
              dic['median_y_a'],
              dic['median_z_a'],
              dic['median_x_g'],
              dic['median_y_g'],
              dic['median_z_g'],
              
              dic['var_x_a'],
              dic['var_y_a'],
              dic['var_z_a'],
              dic['var_x_g'],
              dic['var_y_g'],
              dic['var_z_g'],
              
              dic['aadp_x_a'],
              dic['aadp_y_a'],
              dic['aadp_z_a'],
              dic['aadp_x_g'],
              dic['aadp_y_g'],
              dic['aadp_z_g'],
              
              dic['ptp_x_a'],
              dic['ptp_y_a'],
              dic['ptp_z_a'],
              dic['ptp_x_g'],
              dic['ptp_y_g'],
              dic['ptp_z_g'],
              
              dic['mode_x_a'],
              dic['mode_y_a'],
              dic['mode_z_a'],
              dic['mode_x_g'],
              dic['mode_y_g'],
              dic['mode_z_g'],
              
              dic['cov_x_a'],
              dic['cov_y_a'],
              dic['cov_z_a'],
              dic['cov_x_g'],
              dic['cov_y_g'],
              dic['cov_z_g'],
              
              dic['mad_x_a'],
              dic['mad_y_a'],
              dic['mad_z_a'],
              dic['mad_x_g'],
              dic['mad_y_g'],
              dic['mad_z_g'],
              
              dic['iqr_x_a'],
              dic['iqr_y_a'],
              dic['iqr_z_a'],
              dic['iqr_x_g'],
              dic['iqr_y_g'],
              dic['iqr_z_g'],
              
              dic['correlate_xy_a'],
              dic['correlate_yz_a'],
              dic['correlate_xz_a'],
              dic['correlate_xy_g'],
              dic['correlate_yz_g'],
              dic['correlate_xz_g'],
              
              dic['skew_x_a'],
              dic['skew_y_a'],
              dic['skew_z_a'],
              dic['skew_x_g'],
              dic['skew_y_g'],
              dic['skew_z_g'],
              
              dic['kurtosis_x_a'],
              dic['kurtosis_y_a'],
              dic['kurtosis_z_a'],
              dic['kurtosis_x_g'],
              dic['kurtosis_y_g'],
              dic['kurtosis_z_g'],
              
              dic['spectral_energy_x_a'],
              dic['spectral_energy_y_a'],
              dic['spectral_energy_z_a'],
              dic['spectral_energy_x_g'],
              dic['spectral_energy_y_g'],
              dic['spectral_energy_z_g'],
              
#               dic['spectral_entropy_x_a'],
#               dic['spectral_entropy_y_a'],
#               dic['spectral_entropy_z_a'],
#               dic['spectral_entropy_x_g'],
#               dic['spectral_entropy_y_g'],
#               dic['spectral_entropy_z_g']
             ]
    
    
    return dic, np.array(vector)


# In[ ]:


def rolling_window(a, window, stride):
    shape = a.shape[:-1] + (int((a.shape[-1] - window)/stride + 1), window)
    strides = (stride*a.strides[-1],) + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def getIndices2(sampleSize=1000, step=1000, numSamplePoints=24000):
    indices = np.arange(0, numSamplePoints, 1)
    indices = rolling_window(indices, sampleSize, step)
    
    return indices


def getEncodingArray(df, windows):
    a = []
    for i in range(len(windows)):
        # replaced loc with iloc per documentation
#         a.append(signal_to_encoding(df.loc[windows[i], :])[1])
        a.append(signal_to_encoding(df.iloc[windows[i], :])[1])
        
    return np.array(a)


def deleteDiagonal(array):
    depth = array.shape[-1]
    m = array.shape[1]
    strided = np.lib.stride_tricks.as_strided
    s0,s1,s2 = array.strides
    return strided(array.ravel()[depth:], shape=(m-1, m, depth), strides=(s0+s1,s1, s2)).reshape(m, m-1, depth)


def MinMaxTransformation(windows_features_array):
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(windows_features_array)
#     scaled_array = scaler.transform(windows_features_array)
    
    return scaler
    
    
def getDistFRR1(dfList, window_size = 1000, step = 1000, numSamplePoints= 18000):
    
    windows = getIndices2(sampleSize=window_size, step=step, numSamplePoints= numSamplePoints)

    norm_dist = []
    norm_distro_dict = {}
    counter = 1
    for m in range(len(dfList)):
        
        
        encoding_array = getEncodingArray(dfList[m], windows)
#         print(dfList[m].columns)
#         print(encoding_array.shape)

#         print(np.sum(encoding_array, axis = 1))

        # Doesn't make sense to normalize this here
#         scaler = MinMaxTransformation(encoding_array)
#         encoding_array = scaler.transform(encoding_array)
#         print(encoding_array.shape)
        # Should this be put between braces before indexing??? No
        encoding_array = encoding_array / np.linalg.norm(encoding_array, axis = 1)[:, None]
#         print(np.linalg.norm(encoding_array, axis=1)[:, None].shape)
#         print(encoding_array.shape)
#         print(np.linalg.norm(encoding_array, axis=1))
        # Is dist_array distance calculation done correctly??? It seems it does not square
        # It appears so dist = numpy.linalg.norm(a-b)
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        dist_array = (encoding_array[None, :] - encoding_array[:, None])

        dist_array = deleteDiagonal(dist_array)

        dist_array = np.linalg.norm(dist_array, axis = 2)
        
        norm_dist.append(dist_array)
        
        # for err dist
        norm_distro_dict[m] = np.array(dist_array).ravel()
        
        counter += 1
    
    return {"dist_array": np.array(norm_dist).ravel(), "dist_dict": norm_distro_dict}


def getDistFRRFinal(dfList_exp1, dfList_exp2, window_size = 1000, step = 1000, numSamplePoints= 18000):
    '''
    dfLists are of the same size.
    '''
    if len(dfList_exp1) != len(dfList_exp2): 
        raise Exception("dfLists are not of the same size.")
        
    windows = getIndices2(sampleSize=window_size, step=step, numSamplePoints= numSamplePoints)

    norm_dist = []
    norm_distro_dict = {}
    voting_dist_dict = {}
    counter = 1
    for m in range(len(dfList_exp1)):
        
        
        encoding_array_exp1 = getEncodingArray(dfList_exp1[m], windows)
        encoding_array_exp2 = getEncodingArray(dfList_exp2[m], windows)
        
        # Doesn't make sense to normalize this here
        scaler = MinMaxTransformation(encoding_array_exp1)
        encoding_array_exp1 = scaler.transform(encoding_array_exp1)
#         encoding_array_exp2 = scaler.transform(encoding_array_exp2)
        
#         # approach 2: Not intuitive as you dont have access to all of user2s stream of 4 min, only windowsizes at a time
#         scaler = MinMaxTransformation(encoding_array_exp2)

        encoding_array_exp2 = scaler.transform(encoding_array_exp2)
        
#         print(dfList[m].columns)
#         print(encoding_array.shape)

#         print(np.sum(encoding_array, axis = 1))
#         print((encoding_array_exp1 / np.linalg.norm(encoding_array_exp1, axis = 1)[:, None]).shape)
#         print(encoding_array_exp1.shape)
#         print(encoding_array.shape)
        # Should this be put between braces before indexing??? No
        encoding_array_exp1 = encoding_array_exp1 / np.linalg.norm(encoding_array_exp1, axis = 1)[:, None]
        encoding_array_exp2 = encoding_array_exp2 / np.linalg.norm(encoding_array_exp2, axis = 1)[:, None]
#         print(np.linalg.norm(encoding_array, axis=1)[:, None].shape)
#         print(encoding_array.shape)
#         print(np.linalg.norm(encoding_array, axis=1))



        
        # Is dist_array distance calculation done correctly??? It seems it does not square
        # It appears so dist = numpy.linalg.norm(a-b)
        # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        dist_array = (encoding_array_exp1[None, :] - encoding_array_exp2[:, None])
#         print(dist_array.shape)
        
        dist_array = np.linalg.norm(dist_array, axis = 2)
        
        norm_dist.append(dist_array)
        
#         print(dist_array.shape)
        # for err dist
        norm_distro_dict[m] = np.array(dist_array).ravel()
        
        # for voting dist
        voting_dist_dict[m] = dist_array[None, :]
        
        
        counter += 1
        
#         print(voting_dist_dict[0].shape)
    
    return {"dist_array": np.array(norm_dist).ravel(), "dist_dict": norm_distro_dict, "voting_dist_dict": voting_dist_dict}


def getDistFARFinal(dfList, window_size = 1000, step = 1000, numSamplePoints= 18000):
    
    windows = getIndices2(sampleSize=window_size, step=step, numSamplePoints= numSamplePoints)

    norm_dist = []
    norm_distro_dict = {}
    voting_dist_dict = {}
    counter = 1
    
    encoding_array_dic = {}
    for i in range(len(dfList)):
        encoding_array_dic[i] = getEncodingArray(dfList[i], windows)
        # Should this be put between braces before indexing???
#         # Should this be done here? Not here this only makes the result vector small and should be at the end
#         encoding_array_dic[i] = encoding_array_dic[i] / np.linalg.norm(encoding_array_dic[i], axis = 1)[:, None]
        
    for m in range(len(dfList)):
        
        cum_distro_array = []
        encoding_array_m = encoding_array_dic[m]
        
        # Should I scale the new vector with the transform of the user profile?
        scaler = MinMaxTransformation(encoding_array_m)
        encoding_array_m = scaler.transform(encoding_array_m)
        
        encoding_array_m = encoding_array_m / np.linalg.norm(encoding_array_m, axis = 1)[:, None]
        
        for k in range(len(dfList)):
            
            if m != k:
                
                encoding_array_k = encoding_array_dic[k]
                
#                 # approach 2
#                 scaler = MinMaxTransformation(encoding_array_k)
                
                # Scale array_k with array_m transform
                encoding_array_k = scaler.transform(encoding_array_k)
                
                encoding_array_k = encoding_array_k / np.linalg.norm(encoding_array_k, axis = 1)[:, None]
                
                # Is dist_array distance calculation done correctly??? It seems it does not square. No, resolved
                # It appears so dist = numpy.linalg.norm(a-b)
                # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
                dist_array = (encoding_array_m[None, :] - encoding_array_k[:, None])
                
                # print and check dimensions
                
                dist_array = np.linalg.norm(dist_array, axis = 2)

                norm_dist.append(dist_array)
                
                # for err dist
                cum_distro_array.append(dist_array)
                

        
        norm_distro_dict[m] = np.array(cum_distro_array).ravel()
        
        # for voting dist
        voting_dist_dict[m] = np.array(cum_distro_array)

        counter += 1
        
#         print(voting_dist_dict[0].shape)
    
    return {"dist_array": np.array(norm_dist).ravel(), "dist_dict": norm_distro_dict, "voting_dist_dict": voting_dist_dict}


def decision_confidence(dist_array, dist_threshold):
    '''
    input: dist_array: (N, # unknown_user windows, # genuine_user windows)
    output: (N, de/auth decision percentage)
    '''
    vals = np.where(dist_array < dist_threshold, 1, 0)
    windows_decision_confidence = np.mean(vals, axis = 2)

    return windows_decision_confidence


def decision_module(dist_array, dist_threshold, acceptance_threshold):
    '''
    input: dist_array: (N, # unknown_user windows, # genuine_user windows)
    output: (N, de/auth boolean decision)
    '''
    
    windows_decision_confidence = decision_confidence(dist_array, dist_threshold = dist_threshold)
    windows_final_decision = np.where(windows_decision_confidence >= acceptance_threshold, 1, 0)
    
    return windows_final_decision
    
def FRR_vote_based(dist_array, dist_threshold = None, acceptance_threshold = None):
    '''
    input: dist_array: (N, # unknown_user windows, # genuine_user windows)
    output: vote based FRR
    '''
    
    #good note but not applicable here np.where((a==0)|(a==1), a^1, a)

    windows_final_decision = decision_module(dist_array, dist_threshold, acceptance_threshold)
    vals = windows_final_decision^1
    
    return np.mean(vals)


def FAR_vote_based(dist_array, dist_threshold = None, acceptance_threshold = None):
    '''
    input: dist_array: (M*N, # unknown_user windows, # genuine_user windows)
    output: vote based FAR
    '''
    windows_final_decision = decision_module(dist_array, dist_threshold, acceptance_threshold)
    vals = windows_final_decision
    
    return np.mean(vals)


def FRR(dist, threshold):
    
    vals = np.where(dist < threshold, 0, 1)
    
    return np.mean(vals)


def FAR(dist, threshold):
    
    vals = np.where(dist < threshold, 1, 0)
    
    return np.mean(vals)


def DistroFRR(dist_dict, threshold):
    
    distro = []
    for i in range(len(dist_dict)):
        vals = np.where(dist_dict[i] < threshold, 0, 1)
        distro.append(sum(vals))
        
    return distro


def DistroFAR(dist_dict, threshold):
    
    distro = []
    for i in range(len(dist_dict)):
        vals = np.where(dist_dict[i] < threshold, 1, 0)
        distro.append(sum(vals))
        
    return distro


# In[ ]:


# np.where(np.array([.5, 1]) < .6, 1, 0)
# a = np.array(range(5, 11))
# b = np.array(range(2, 6))

# res = a[None, :] - b[:, None]
# # print(res)
# a = res % 2
# print(a)
# a^1
# b = np.array([a, a-9])
# print(b)
# np.concatenate(b)


# In[ ]:


# d = {}
# d[0] = np.array([[[1,2]]])
# d[1] = np.array([[[2,3]]])
# a = np.array(list(d.values()))
# print(a.shape)
# a = np.concatenate(a)
# print(a.shape)
# a = np.concatenate(a)
# print(a.shape)


# In[ ]:


def getEER(distFRR, distFAR, thresholdList=None):
    
    if thresholdList is None:
        thresholdList = np.arange(0, 3, 0.001)
    
    farList = []
    frrList = []
    
    eer = []
    for t in thresholdList:
        far = FAR(distFAR, threshold = t)
        frr = FRR(distFRR, threshold = t)
        farList.append(far)
        frrList.append(frr)
        eer.append(abs(far-frr))
        
    eer = np.array(eer)
    eer[eer==0] = 99999
    print(farList[np.argmin(eer)])
    print(frrList[np.argmin(eer)])

#     print("EER: {}".format((frrList[np.argmin(eer)] + farList[np.argmin(eer)])/2))
    return {"EER": (frrList[np.argmin(eer)] + farList[np.argmin(eer)])/2, "farList": farList, "frrList": frrList, "EER_threshold": thresholdList[np.argmin(eer)]}


# In[ ]:


def getEERVoteBased(dist_array_FRR, dist_array_FAR, thresholdList=None):
    
    if thresholdList is None:
        thresholdList = np.arange(0, 3, 0.001)
    
    farList = []
    frrList = []
    
    eer = []
    acceptance_threshold = .6
    
    for t in thresholdList:
        far = FAR_vote_based(dist_array_FAR, dist_threshold = t, acceptance_threshold = acceptance_threshold)
        frr = FRR_vote_based(dist_array_FRR, dist_threshold = t, acceptance_threshold = acceptance_threshold)
        farList.append(far)
        frrList.append(frr)
        eer.append(abs(far-frr))
        
    eer = np.array(eer)
    eer[eer==0] = 99999
    print(farList[np.argmin(eer)])
    print(frrList[np.argmin(eer)])

#     print("EER: {}".format((frrList[np.argmin(eer)] + farList[np.argmin(eer)])/2))
    return {"EER": (frrList[np.argmin(eer)] + farList[np.argmin(eer)])/2, "farList": farList, "frrList": frrList, "EER_threshold": thresholdList[np.argmin(eer)]}


# In[ ]:


def getEERWindowsDict(dfList_exp1, start_window_size=250, end_window_size=3000, increment_step=250, numSamplePoints=22001, isEqualSampleSize = False, fixedSampleStep=3000, thresholdList=None, dfList_exp2=None):
    
    window_EER_dict = {}
    window_EER_threshold_dict = {}
    window_farList_dict = {}
    window_frrList_dict = {}
    window_farDistro_dict = {}
    window_frrDistro_dict = {}
    
    lst = np.arange(start_window_size, end_window_size + 1, increment_step)
    
    for w in lst:
        if isEqualSampleSize:
            sampleStep = fixedSampleStep
        else:
            sampleStep = w
        
        if dfList_exp2 is None:
            print("dfList_exp2 is None")
            distFRRDATA = getDistFRR1(dfList_exp1, window_size = w, step = sampleStep, numSamplePoints= numSamplePoints)
        else:
            print("dfList_exp2 is Not None")
            distFRRDATA = getDistFRRFinal(dfList_exp1, dfList_exp2, window_size = w, step = sampleStep, numSamplePoints= numSamplePoints)
            
        distFARDATA = getDistFARFinal(dfList_exp1, window_size = w, step = sampleStep, numSamplePoints= numSamplePoints)
        
#         print('--- start of voting based')
#         print(distFRRDATA["voting_dist_dict"][0].shape)
#         print(distFARDATA["voting_dist_dict"][0].shape)
#         print(np.concatenate(list(distFRRDATA["voting_dist_dict"].values())).shape)
#         print(np.concatenate(list(distFARDATA["voting_dist_dict"].values())).shape)
        
        
#         voting_dist_FRR = np.concatenate(list(distFRRDATA["voting_dist_dict"].values()))
#         voting_dist_FAR = np.concatenate(list(distFARDATA["voting_dist_dict"].values()))
#         voting_EER_data = getEERVoteBased(voting_dist_FRR, voting_dist_FAR, thresholdList=thresholdList)
        
#         print("numParticipants: {}, windowSize: {}, isEqualSampleSize: {}, EER: {}".format(len(dfList_exp1), w, isEqualSampleSize, voting_EER_data["EER"]))
        
#         print("--- end of voting based")
        
        distFRR = distFRRDATA["dist_array"]
        distFAR = distFARDATA["dist_array"]
        EER_data = getEER(distFRR, distFAR, thresholdList=thresholdList)
        
        window_EER_dict[w] = EER_data["EER"]
        window_EER_threshold_dict[w] = EER_data["EER_threshold"]
        window_farList_dict[w] = EER_data["farList"]
        window_frrList_dict[w] = EER_data["frrList"]
        window_farDistro_dict[w] = distFARDATA["dist_dict"]
        window_frrDistro_dict[w] = distFRRDATA["dist_dict"]
        
        
        
        print("numParticipants: {}, windowSize: {}, isEqualSampleSize: {}, EER: {}".format(len(dfList_exp1), w, isEqualSampleSize, window_EER_dict[w]))
        
    return { "window_EER_dict": window_EER_dict, "window_EER_threshold_dict": window_EER_threshold_dict , "window_farList_dict": window_farList_dict, "window_frrList_dict": window_frrList_dict, "window_farDistro_dict": window_farDistro_dict, "window_frrDistro_dict": window_frrDistro_dict}


# In[ ]:


def getErrFixedThreshold(distFRR, distFAR, threshold):
        
    far = FAR(distFAR, threshold = threshold)
    frr = FRR(distFRR, threshold = threshold)
    

    return {"FAR": far, "FRR": frr, "threshold": threshold}


# In[ ]:


# To be done
def getErrFixedThresholdWindowsDict(dfList, start_window_size=250, end_window_size=3000, increment_step=250, numSamplePoints=22001, isEqualSampleSize = False, fixedSampleStep=3000):
    
    window_EER_dict = {}
    window_EER_threshold_dict = {}
    window_farDistro_dict = {}
    window_frrDistro_dict = {}
    
    lst = np.arange(start_window_size, end_window_size + 1, increment_step)
    
    for w in lst:
        if isEqualSampleSize:
            sampleStep = fixedSampleStep
        else:
            sampleStep = w
            
        distFRRDATA = getDistFRRFinal(dfList, window_size = w, step = sampleStep, numSamplePoints= numSamplePoints)
        distFARDATA = getDistFARFinal(dfList, window_size = w, step = sampleStep, numSamplePoints= numSamplePoints)
        
        distFRR = distFRRDATA["dist_array"]
        distFAR = distFARDATA["dist_array"]
        EER_data = getEER(distFRR, distFAR)

        
        window_EER_dict[w] = EER_data["EER"]
        window_EER_threshold_dict[w] = EER_data["EER_threshold"]
        window_farDistro_dict[w] = distFARDATA["dist_dict"]
        window_frrDistro_dict[w] = distFRRDATA["dist_dict"]
        
        print("numParticipants: {}, windowSize: {}, isEqualSampleSize: {}, EER: {}".format(len(dfList), w, isEqualSampleSize, window_EER_dict[w]))
        
    return { "window_EER_dict": window_EER_dict, "window_EER_threshold_dict": window_EER_threshold_dict , "window_farList_dict": window_farList_dict, "window_frrList_dict": window_frrList_dict, "window_farDistro_dict": window_farDistro_dict, "window_frrDistro_dict": window_frrDistro_dict}


# In[ ]:


# cite the thesis paper i found
def utils_ppp(P):
    """Pretty print parameters of an experiment."""
    df = pd.DataFrame([asdict(P)])
    df = df.T
    df.columns = ["Value"]
    
    display(df)


# In[ ]:


# source: https://github.com/dynobo/ContinAuth/blob/master/notebooks/utils.ipynb
def utils_eer(y_true, y_pred, return_threshold=False):
    """Calculate the Equal Error Rate.

    Based on https://stackoverflow.com/a/49555212, https://yangcha.github.io/EER-ROC/
    and https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object

    Arguments:
        y_true {np.array}  -- Actual labels
        y_pred {np.array}  -- Predicted labels or probability
        
    Returns:
        float              -- Equal Error Rate        
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)  # Calculated threshold, not needed for score
    if return_threshold:
        return eer, thresh
    else:
        return eer


# In[ ]:


if TEST_MODE:
    temp_eer, tres = utils_eer(
        [-1, -1, -1, 1, 1], [0, 0.9, 0.1, 0.74, 0.8], return_threshold=True
    )
    print(f"EER: {temp_eer:.3f}, Threshold: {tres:.3f} <-- Arbitrary case")

    temp_eer, tres = utils_eer(
        [-1, -1, -1, 1, 1], [0.1, 0.2, 0.3, 1, 0.9], return_threshold=True
    )
    print(f"EER: {temp_eer:.3f}, Threshold: {tres:.3f} <-- Best case")

    temp_eer, tres = utils_eer(
        [1, 1, 1, -1, -1], [0.1, 0.2, 0.3, 1, 0.9], return_threshold=True
    )
    print(f"EER: {temp_eer:.3f}, Threshold: {tres:.3f} <-- Worse case")
    
#     new case does it make sense? I don't think so
    temp_eer, tres = utils_eer(
        [1, 1, 1, -1, -1], [-1, 1, -1, -1, -1], return_threshold=True
    )
    print(f"EER: {temp_eer:.3f}, Threshold: {tres:.3f} <-- Worse case")


# In[ ]:


utils_eer_scorer = make_scorer(utils_eer, greater_is_better=False)


# # Split Dataset for Valid/Test  
# In two splits: one used during hyperparameter optimization, and one used during testing.
# 
# The split is done along the subjects: All sessions of a single subject will either be in the validation split or in the testing split, never in both.
# 
# They did a 30 60 split.

# # Reshaping Raw Features.
# We have our own function of windows for this. Do this for both training and testing.
# 
# # Extracting time and frequency based features.
# Again, we have a function for this. Do this for both training and testing.

# # Hyperparameter Optimization 
# 
# I do not find any reaqsonable explaination how to use a cross-validation as we are talking about anomaly detection.
# 
# I am using the experiment 1 data as train, and experiment 2 data as validation.

# # Using SVM in a real-world Scenario with multiple genuine users and intruders
# Source: https://datascience.stackexchange.com/questions/23623/what-is-the-best-way-to-classify-data-not-belonging-to-set-of-classes
# 
# Stage 1: 
#     Use one-class SVM to assign those images that do not belong to the set of predefined classes as the 9-th class.
# 
# Stage 2:
#     For those images that passes through your filter, let the multi-class SVM assign them to one of the 8 classes.

# Loading data:

# In[ ]:


def load_data_frames(user_ids, begin_idx, end_idx, min_len):
    '''
    input: 
        user_ids: list of approved user_ids after exploratory data analysis
        begin_idx: the index before which data is discarded for user i
        end_idx: the index after which data is discarded for user i
        min_len: the minimum length that a dataframe has to be after cutting of both endings
        
    output:
        {dfList_exp1, dfList_exp2}: return dfList for exp1 and exp2 of the selected user_ids
    '''
    print("Loading exp1 data:")
    dfList_exp1 = []
    for i in user_ids:
        dic = getDataStats1(i, begin_idx=begin_idx, end_idx=end_idx)

        if(dic['accel']<min_len):
            raise Exception("The Stream is shorter than {}".format(min_len))

        dfList_exp1 = dfList_exp1 + [dic['df'].reset_index(drop=True)]


    print("Loading exp2 data:")
    dfList_exp2 = []
    for i in user_ids:
        dic = getDataStats2(i, begin_idx=begin_idx, end_idx=end_idx)

        if(dic['accel']<min_len):
            raise Exception("The Stream is shorter than {}".format(min_len))

        dfList_exp2 = dfList_exp2 + [dic['df'].reset_index(drop=True)]
    #     dfList = dfList + [dic['df']]
    
    return {"dfList_exp1": dfList_exp1, "dfList_exp2": dfList_exp2}


# In[ ]:


def MakeXExpDic(dfList_exp1, dfList_exp2, window_size = 1000, step = 1000, numSamplePoints= 18000):
    '''
    return 
    X_exp1_dic
    X_exp2_dic
    dfLists are of the same size.
    '''
    if len(dfList_exp1) != len(dfList_exp2): 
        raise Exception("dfLists are not of the same size.")
    
    windows = getIndices2(sampleSize=window_size, step=step, numSamplePoints= numSamplePoints)

    X_exp1_dic = {}
    X_exp2_dic = {}
    for i in range(len(dfList_exp1)):
        
        
        encoding_array_exp1 = getEncodingArray(dfList_exp1[i], windows)
        encoding_array_exp2 = getEncodingArray(dfList_exp2[i], windows)
        
        X_exp1_dic[i] = encoding_array_exp1
        X_exp2_dic[i] = encoding_array_exp2
        
    return {"X_exp1_dic": X_exp1_dic, "X_exp2_dic": X_exp2_dic}


def OneClassSVMSets(k, X_exp1_dic, X_exp2_dic, cv=5):
    '''
    return the required sets for an OCSVM trained on the user with key. 
    X_train: X data from X_exp1_dic[k]
    X_test_regular: X data from X_exp2_dic[k]
    X_test_anomalous: X data from X_exp2_dic[!k]
    '''
    
    if k not in  X_exp1_dic:
        raise Exception("invalid key for dic")
        
    
    X_pos = X_exp1_dic[k]
#     X_neg = np.concatenate([X_exp1_dic[key] for key in X_exp1_dic.keys() if key != k], axis=0)
    X_test_regular = X_exp2_dic[k]
    X_test_anomalous = np.concatenate([X_exp2_dic[key] for key in X_exp2_dic.keys() if key != k], axis=0)
    
    
#     n, m = len(Xpos), len(Xneg)
    np.random.shuffle(X_neg)
    print((X_neg.shape[0], X_pos.shape[0]))
    X_neg = X_neg[np.random.choice(X_neg.shape[0], size=X_pos.shape[0], replace=False), :]
    print(X_pos.shape, X_neg.shape)
    # Creating (train, test) tuples of indices for k-folds cross-validation
    # We split the positive class (normal data) as we only want the positive examples in the training set.
    
    train_splits = KFold(n_splits=cv, shuffle=True).split(X_pos)
    anomalous_splits = KFold(n_splits=cv, shuffle=True).split(X_neg)

#     print(len(train_splits), len(anomalous_splits))
    # Negative examples (abnormal data) are added to the test set (see https://stackoverflow.com/a/58459322/3673842)
    y_train = np.concatenate([np.repeat(1.0, len(X_pos)), np.repeat(-1.0, len(X_neg))])
    X_train = np.concatenate([X_pos, X_neg], axis=0)
    
    # https://github.com/steppi/adeft/blob/anomaly_detection/adeft/modeling/find_anomalies.py#L170
    cv_splits = ((train, np.concatenate((test, anom_test + X_pos.shape[0]), axis = 0))
                  for (train, test), (_, anom_test)
                  in zip(train_splits, anomalous_splits))
    
    return {"X_train": X_train, "y_train": y_train, "X_test_regular": X_test_regular, "X_test_anomalous": X_test_anomalous, "cv_splits": cv_splits}


# # From CNN file

# In[ ]:


def getRawDataChunks(df, windows, scale=True, scaler="MinMaxScaler", user_idx=None, exp_num=None):
    a = []
    df = df.drop(columns=["time_stamp"]).copy()
    df_array = df.to_numpy()
    
    if scale:
        print(f"user_idx: {user_idx}, exp_num: {exp_num}, scale: {scale}, scaler: {scaler}")
        print(df_array.shape)
        scaler = get_new_scaler_dict[scaler]
        scaler = scaler().fit(df_array)
        df_array = scaler.transform(df_array)
        scaled_df = pd.DataFrame(data=df_array, columns = df.columns, dtype=df_array.dtype)
        df = scaled_df
    
    for i in range(len(windows)):
        # replaced loc with iloc per documentation
#         a.append(df_array[windows[i], :]) #CNN
        a.append(df.iloc[windows[i], :]) #waca
    
#     print(len(a))
#     print(len(a))
#     return np.array([a])
#     return np.array(a), scaler #CNN
    return a, scaler #waca

def MakeRawXExpDic(dfList_exp1, dfList_exp2, window_size = 1000, step = 1000, numSamplePoints= 18000, scale_exp1=False, scale_exp2=True, scaler="MinMaxScaler"):
    '''
    return 
    X_exp1_dic
    X_exp2_dic
    dfLists are of the same size.
    '''

    if len(dfList_exp1) != len(dfList_exp2): 
        raise Exception("dfLists are not of the same size.")
    
    windows = getIndices2(sampleSize=window_size, step=step, numSamplePoints= numSamplePoints)

    X_exp1_dic = {}
    X_exp2_dic = {}
    fitted_scaler_exp1_dic={}
    fitted_scaler_exp2_dic={}
    for i in range(len(dfList_exp1)):
        
        
        encoding_array_exp1, fitted_scaler_exp1 = getRawDataChunks(dfList_exp1[i], windows, scale=scale_exp1, scaler=scaler, user_idx=i, exp_num=1)
        encoding_array_exp2, fitted_scaler_exp2 = getRawDataChunks(dfList_exp2[i], windows, scale=scale_exp2, scaler=scaler, user_idx=i, exp_num=2)
        
        X_exp1_dic[i] = encoding_array_exp1
        X_exp2_dic[i] = encoding_array_exp2
        
        fitted_scaler_exp1_dic[i]=fitted_scaler_exp1
        fitted_scaler_exp2_dic[i]=fitted_scaler_exp2
        
    return {"Raw_X_exp1_dic": X_exp1_dic, "Raw_X_exp2_dic": X_exp2_dic, "fitted_scaler_exp1_dic": fitted_scaler_exp1_dic, "fitted_scaler_exp2_dic": fitted_scaler_exp2_dic}



def MakeDeepXExpDic(dfList_exp, deep_feature_model, fitted_scaler_dic=None):
    '''
    ???
    return 
    X_exp_dic
    dfLists are of the same size.
    '''

    X_exp_dic = {}
    for k in dfList_exp.keys():
        if fitted_scaler_dic:
            print(f"scaling exp1 samples of user: {k}")
            X_exp_dic[k] = deep_feature_model.predict(transform_user_windows(dfList_exp[k], fitted_scaler_dic[k]))
        else:
            print(f"not scaling exp2 samples of user: {k}")
            X_exp_dic[k] = deep_feature_model.predict(dfList_exp[k])
        
        
    return X_exp_dic



def MakeWACAXExpDic(dfList_exp, fitted_scaler_dic=None):
    '''
    ???
    return 
    X_exp_dic
    dfLists are of the same size.
    '''

    X_exp_dic = {}
    for k in dfList_exp.keys():
        if fitted_scaler_dic:
            print(f"scaling exp1 samples of user: {k}")
            X_exp_dic[k] = ExtractWACAFeatures(transform_user_windows(dfList_exp[k], fitted_scaler_dic[k]))
        else:
            print(f"not scaling exp2 samples of user: {k}")
            X_exp_dic[k] = ExtractWACAFeatures(dfList_exp[k])
        
        
    return X_exp_dic
    

def ExtractWACAFeatures(X_exp):
    a = []
    for window in X_exp:
        a.append(signal_to_encoding(window)[1])
        
    return np.array(a)


def MakeScaledXExpDic(df_exp_dict, fitted_scaler_dic):
    '''
    ???
    return 
    X_exp_dic
    dfLists are of the same size.
    '''

    X_exp_dic = {}
    for k in df_exp_dict.keys():
        print(f"scaling exp1 samples of user: {k}")
        X_exp_dic[k] = transform_user_windows(df_exp_dict[k], fitted_scaler_dic[k])
        
        
    return X_exp_dic


def scale_feature_windows(df_exp_dict, fitted_scaler_dic=None, scaler_type=None, scaler_clip=False):
    '''
    ???
    return 
    X_exp_dic
    dfLists are of the same size.
    '''
    if fitted_scaler_dic == None:
        fitted_scaler_dic={}
        
    X_exp_dic = {}
    for k in df_exp_dict.keys():
        if k in fitted_scaler_dic:
            print(f"transform exp1 samples of user: {k}")
        else:
            print(f"fit_transform exp2 samples of user: {k}")
            print(f"user_idx: {k}, exp_num: {2}, scaler: {scaler_type}, scaler_clip: {scaler_clip}")
            scaler = get_new_scaler_dict[scaler_type]
            scaler = scaler(clip=scaler_clip).fit(df_exp_dict[k])
            fitted_scaler_dic[k] = scaler

#         print(df_exp_dict[k].shape)
        X_exp_dic[k] = transform_user_WACA_windows(df_exp_dict[k], fitted_scaler_dic[k])
        
        
    return {"X_exp_dic": X_exp_dic, "fitted_scaler_dic": fitted_scaler_dic}


def transform_user_WACA_windows(X_exp, fitted_scaler):
    
    
    transformed_X_exp = []
    
#     print(X_exp[0].shape)
    for window in X_exp:
        if len(window.shape) == 1:
            window = window.reshape(1, -1)
        scaled_array = fitted_scaler.transform(window)
        transformed_X_exp.append(scaled_array.reshape(-1))
        
    return np.array(transformed_X_exp)


def transform_user_windows(X_exp, fitted_scaler):
    
    
    transformed_X_exp = []
    
#     print(X_exp[0].shape)
    for window in X_exp:
        scaled_array = fitted_scaler.transform(window)
        scaled_window_df = pd.DataFrame(data=scaled_array, columns = window.columns, dtype=scaled_array.dtype)
        transformed_X_exp.append(scaled_window_df)
        
    return transformed_X_exp


# In[ ]:


np.random.choice(range(5), 5, replace = False)


# # utils_plot_distance_hist() For CNN

# In[ ]:


def utils_plot_distance_hist(dist_positive, dist_negative, thres, desc, fig_size=(12, 4), margin=None):
    """Plot histogramm of Euclidean Distances for Positive & Negative Pairs."""

    warnings.filterwarnings("ignore")

    # Plot Distributions
    plt.figure(figsize=fig_size, dpi=180)
    bins = np.linspace(
        min(dist_positive.min(), dist_negative.min()),
        max(dist_positive.max(), dist_negative.max()),
        num=21,
    )
    g1 = sns.distplot(
        dist_positive,
        label="positive pairs",
        bins=bins,
        axlabel=False,
        hist_kws=dict(edgecolor="k", lw=0.5),
        kde_kws=dict(linewidth=0.8),
        color="tab:blue",
    )
    g2 = sns.distplot(
        dist_negative,
        label="negative pairs",
        bins=bins,
        hist_kws=dict(edgecolor="k", lw=0.5),
        kde_kws=dict(linewidth=0.8),
        color="tab:gray",
    )

    # Plot vertical lines
    if thres > 0:
        max_y = max(g1.get_ylim()[1], g2.get_ylim()[1])
        plt.axvline(x=thres, color=MAGENTA, linestyle="--", lw=0.8, alpha=0.7)
        plt.text(
            x=thres + 0.001,
            y=max_y * 0.65,
            s=f"EER Threshold\n({thres:.2f})",
            color=MAGENTA,
            weight="bold",
            fontsize=5,
            alpha=1
        )
        if margin:
            plt.axvline(x=margin, color=MAGENTA, linestyle="--", lw=0.8, alpha=0.7)
            plt.text(
                x=margin + 0.001,
                y=max_y * 0.15,
                s=f"Margin\n({margin})",
                color=MAGENTA,
                weight="bold",
                fontsize=5,
                alpha=1
            )

    # Legend
    plt.legend(
        loc="upper right",
        title=f"{desc} Distances",
        title_fontsize=5,
        fontsize=6,
    )

    warnings.filterwarnings("default")
    return plt


# In[ ]:


if TEST_MODE:
    dist_pos = np.array([0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.8])
    dist_neg = np.array([0.4, 0.5, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 1, 1])
    utils_plot_distance_hist(
        dist_pos, dist_neg, thres=0.4, desc="Pair", fig_size=(12, 4), margin=0.8
    )


# # utils_create_cv_splits()

# In[ ]:


def utils_create_cv_splits(owner_key, train_dic, valid_test_dic, seed=0):
    '''
    return the required sets for an OCSVM trained on the user with key. 
    X_train: X data from train_dic[k], comes from exp2
    X_test_regular: X data from valid_test_dic[k], comes from exp1
    X_test_anomalous: X data from valid_test_dic[!k], comes from exp1
    
    Create cross-validation mask with train-valid pairs.
    
    See e.g. https://stackoverflow.com/a/37591377
    
    Arguments:
        cv_mask {np.ndarray} --
        
    Return:
        {list} -- List of tuple: (<train indices>, <valid indices>)
        
    '''
    
    if owner_key not in  train_dic:
        raise Exception("invalid key for dic")
    
        
    X_pos = train_dic[owner_key].copy()
    X_test_regular = valid_test_dic[owner_key].copy()
    X_test_anomalous = np.concatenate([valid_test_dic[key] for key in valid_test_dic.keys() if key != owner_key], axis=0).copy()
    
    train_idx_owner = np.arange(X_pos.shape[0])
    valid_idx_owner = np.arange(X_test_regular.shape[0]) + train_idx_owner.shape[0]
    
    print(f"owner: {owner_key} train_idx range: {train_idx_owner[0]}, {train_idx_owner[-1]}")
    print(f"owner: {owner_key} valid_idx range: {valid_idx_owner[0]}, {valid_idx_owner[-1]}")
    np.random.seed(seed + owner_key)
    np.random.shuffle(train_idx_owner)
    np.random.shuffle(valid_idx_owner)

    
    cv_splits = []
    base_idx = train_idx_owner.shape[0] + valid_idx_owner.shape[0]
    for key in valid_test_dic.keys():
        
        if key != owner_key:
            # Impostor validation indices
            valid_idx_impostor = np.arange(valid_test_dic[key].shape[0]) + base_idx
            print(f"imposter: {key} valid_idx range: {valid_idx_impostor[0]}, {valid_idx_impostor[-1]}")

            # Balance classes
            min_samples = min(valid_idx_owner.shape[0], valid_idx_impostor.shape[0])
            np.random.seed(seed + key)
            valid_idx_owner_samp = np.random.choice(
                valid_idx_owner, size=min_samples, replace=False
            )
            np.random.seed(seed + key)
            valid_idx_impostor_samp = np.random.choice(
                valid_idx_impostor, size=min_samples, replace=False
            )

            # Concat owner & impostor validation indices
            valid_idx_both = np.hstack([valid_idx_owner_samp, valid_idx_impostor_samp])

            # Add train/valid pair to cv
            cv_splits.append((list(train_idx_owner), list(valid_idx_both)))
            
            base_idx += valid_idx_impostor.shape[0]


    

    y_train = np.concatenate([np.repeat(1.0, X_pos.shape[0]), np.repeat(1.0, X_test_regular.shape[0]), np.repeat(-1.0, X_test_anomalous.shape[0])])
    X_train = np.concatenate([X_pos, X_test_regular, X_test_anomalous], axis=0)
    
    
    return {"X_train": X_train, "y_train": y_train, "X_test_regular": X_test_regular, "X_test_anomalous": X_test_anomalous, "cv_splits": cv_splits}


# In[ ]:


# if TEST_MODE:
#     # Mask Explained:
#     # -2 => Training data (owner)
#     # -1 => Validation data (owner)
#     # 0+ => Validation impostors
#     #              Indices:    0   1   2   3   4   5  6  7  8  9  10 11 12 13 14 15
#     dummy_cv_mask = np.array([-2, -2, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, -2])

#     # Generate tuples of training data and validation data, one tuple for each impostor (0, 1, 2).
#     # Training data (1st list in tuple) contains only indices of owner training data (-2)
#     # Validation data (2nd list in tuple) contains  indices of validation data from owner (-1) and
#     # from a single impostor (0+), each 50 %
#     splits = utils_create_cv_splits(dummy_cv_mask, seed=123)
#     [print(s) for s in splits]


# # utils_cv_report()

# In[ ]:


TEST_MODE=0


# In[ ]:


def utils_cv_report(random_search, owner, impostors):
    """Transform the random_search.cv_results_ into nice formatted dataframe."""
    # Create report
    df_report = pd.DataFrame(random_search.cv_results_)

    # Add owner information
    df_report["owner"] = owner

    # Drop uninteressting columns
    drop_columns = [col for col in df_report.columns if "_train_" in col]
    drop_columns = drop_columns + [col for col in df_report.columns if col.startswith("split") and (col.endswith("recall") or col.endswith("precision") or col.endswith("f1") or col.endswith("roc_auc"))]
    drop_columns = drop_columns + ["params"]
    df_report = df_report.drop(columns=drop_columns)

    # Flip sign of eer (revert flip by sklearn scorer)
    eer_columns = [col for col in df_report.columns if col.endswith("_eer")]
    df_report[eer_columns] = df_report[eer_columns].abs()
    
    # Rename split result columns with impostor-ids used in split
    rename_cols = {}
    for idx, impostor in enumerate(impostors):
        print(f"idx: {idx}, impostor: {impostor}")
        to_rename_cols = [col for col in df_report.columns if col.startswith(f"split{idx}")]
        for col in to_rename_cols:
            rename_cols[col] = str(impostor)+col[len(f"split{idx}"):]
    df_report = df_report.rename(columns=rename_cols)      

    return df_report


# In[ ]:


if TEST_MODE:
    print("Performing Dummy RandomSearch...")
    from sklearn import svm, datasets
    from sklearn.model_selection import RandomizedSearchCV

    iris = datasets.load_iris()
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 2, 3, 4, 5, 6, 7, 10]}
    svc = svm.SVC(gamma="scale")
    clf = RandomizedSearchCV(svc, parameters, cv=3, iid=False)
    clf.fit(iris.data, iris.target)
    print("Create report:")
    df_temp = utils_cv_report(clf, "owner x", ["impo_1", "impo_2", "impo_3"])
    display(df_temp)


# In[ ]:


def utils_plot_randomsearch_results(df_results, n_top=1):
    # Prepare data for plotting
    df_plot = df_results[df_results["rank_test_eer"] <= n_top].rename(
        columns={
            "param_model__nu": r"$\nu$",
            "param_model__gamma": r"$\gamma$",
            "mean_test_accuracy": "Mean Test Acc.",
            "mean_test_eer": "Mean Test EER",
        }
    )
    df_plot["Mean Test EER"] = df_plot["Mean Test EER"] * -1  # Because fewer is more

    median_nu = df_plot[r"$\nu$"].median()
    median_gamma = df_plot[r"$\gamma$"].median()

    # Plot
    fig = plt.figure(figsize=(5.473 / 1.3, 2), dpi=180)
    g = sns.scatterplot(
        x=r"$\nu$",
        y=r"$\gamma$",
        data=df_plot,
        size="Mean Test EER",
        sizes=(7, 60),
        hue="Mean Test EER",
        alpha=1,
        #        palette="Blues",
        linewidth=0,
    )

    # Format Legend labels
    leg = g.get_legend()
    new_handles = [h for h in leg.legendHandles]
    new_labels = []
    for i, handle in enumerate(leg.legendHandles):
        label = handle.get_label()
        print(f'{i}, {label}')
        if ord(label[0]) == 8722:
            label = '-' + label[1:]
            
        if i != 0:
            
            try:
                new_labels.append(f"{abs(float(label)):.3f}")

            except ValueError:
                new_labels.append("")

    # Plot mean values
    plt.plot(
        [-0.01, 0.31],
        [median_gamma, median_gamma],
        linestyle="dashed",
        linewidth=0.8,
        alpha=0.7,
        color="black",
    )
    plt.text(
        0.23,
        median_gamma * 1.7 ** 2,
        r"median($\gamma$)",
        fontsize=6,
        color="black",
        alpha=0.9,
    )
    plt.text(
        0.23,
        median_gamma * 1.2 ** 2,
        f"{median_gamma:.3f}",
        fontsize=5,
        color="black",
        alpha=0.9,
    )

    plt.plot(
        [median_nu, median_nu],
        [0.0001, 1000],
        linestyle="dashed",
        linewidth=0.8,
        alpha=0.7,
        color="black",
    )
    plt.text(
        median_nu + 0.005, 400, r"median($\nu$)", fontsize=6, color="black", alpha=0.9
    )
    plt.text(
        median_nu + 0.005, 200, f"{median_nu:.3f}", fontsize=5, color="black", alpha=0.9
    )

    # Adjust axes & legend
    plt.yscale("log")
    plt.ylim(0.0001, 1000)
    plt.xlim(0, 0.305)
#     print(new_handles)
    print(new_labels)
    plt.legend(
        new_handles,
        new_labels,
        bbox_to_anchor=(1.02, 1),
        loc=2,
        borderaxespad=0.0,
        title="Mean EER per Owner\n(Validation Results)",
        title_fontsize=5,
    )

    fig.tight_layout()
    return median_nu, median_gamma, fig


# In[ ]:





# In[ ]:


def utils_plot_acc_eer_dist(df_plot, y_col):
    n_subject = len(df_plot["Owner"].unique()) - 1
    mean_col = df_plot[y_col].mean()

    fig = plt.figure(figsize=(5.473, 2), dpi=180)
    ax = sns.boxplot(x="Owner", y=y_col, data=df_plot, **utils_boxplot_style)
    ax.set_ylim((0, 1))

    plt.plot(
        [-0.6, n_subject + 0.6],
        [mean_col, mean_col],
        linestyle="dashed",
        linewidth=1,
        color=MAGENTA,
        alpha=0.7,
    )
    plt.text(n_subject + 0.6, mean_col, f"mean", fontsize=6, color=MAGENTA)
    plt.text(
        n_subject + 0.6, mean_col - 0.04, f"{mean_col:.3f}", fontsize=4.5, color=MAGENTA
    )
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    print(f"Overall mean: {mean_col:.4f}")
    return fig


# In[ ]:





# In[ ]:


# if TEST_MODE:
#     print("Performing Dummy RandomSearch...")
#     from sklearn import svm, datasets
#     from sklearn.model_selection import RandomizedSearchCV

#     iris = datasets.load_iris()
#     parameters = {"kernel": ("linear", "rbf"), "C": [1, 2, 3, 4, 5, 6, 7, 10]}
#     svc = svm.SVC(gamma="scale")
#     clf = RandomizedSearchCV(svc, parameters, cv=3, iid=False)
#     clf.fit(iris.data, iris.target)
#     print("Create report:")
#     df_temp = utils_cv_report(clf, "owner x", ["impo_1", "impo_2", "impo_3"])
#     display(df_temp)


# In[ ]:


class pca_feature_selector:
    def __init__(self, n_components):
        self._pca_dict = {}
        self.n_components = n_components
        
    def add_user_pca(self, owner_idx, user_pca):
        if owner_idx in self._pca_dict:
            raise Exception(f"owner_idx: {owner_idx} alraedy exists!")
        
        self._pca_dict[owner_idx] = user_pca
        
    def user_feature_ranking(self, owner_idx):
        '''
        these two are the same 
        np.matmul(pca.explained_variance_ratio_[np.newaxis], abs_components) == np.dot(pca.explained_variance_ratio_, abs_components)[np.newaxis]
        '''
        pca = self._pca_dict[owner_idx]
        abs_components = np.abs(pca.components_)
        feature_importance = np.dot(pca.explained_variance_ratio_, abs_components)[np.newaxis]
        top_feature_indices = np.argsort(-1*feature_importance)
        
        return {"top_feature_indices": top_feature_indices, "feature_importance": feature_importance}
    
    def get_comparison_matrix(self):
        
        feature_importance_matrix = []
        top_feature_matrix = []
        for owner_idx in self._pca_dict:
            user_feature_dict = self.user_feature_ranking(owner_idx) 
            feature_importance_matrix += [user_feature_dict["feature_importance"]]
            top_feature_matrix += [user_feature_dict["top_feature_indices"]]
            
        self._feature_importance_matrix = np.concatenate(feature_importance_matrix, axis=0)
        self._top_feature_matrix = np.concatenate(top_feature_matrix, axis=0)

        return {"feature_importance_matrix": self._feature_importance_matrix, "top_feature_matrix" :self._top_feature_matrix}
    
    def find_top_n_features(self):
        
        best_feature_lst = []
        for i in range(self.n_components):
            best_feature_lst.append(self.find_next_best_feature(best_feature_lst))
            
        return best_feature_lst
        
    def find_next_best_feature(self, curr_feature_lst):
        
        curr_pc_idx = len(curr_feature_lst)
        feature_column_count = np.bincount(self._top_feature_matrix[:, curr_pc_idx])
        print(f"top_f_m: {self._top_feature_matrix[:, curr_pc_idx]}")
        print(curr_feature_lst)
        
        i = 0
        #probably need to use a tree type or heap structure
        while i < len(feature_column_count):
            top_feature_idx = np.argmax(feature_column_count[i:]) + i
            if top_feature_idx not in curr_feature_lst:
                return top_feature_idx
            print('-------')
            print(feature_column_count)
            print(f"i: {i}, top_feature_idx: {top_feature_idx}")

            i = top_feature_idx + 1
            
        raise Exception('could not find best feature')


# In[ ]:


def utils_plot_training_loss(history):
    """Plot Train/Valid Loss during Epochs."""
    fig = plt.figure(figsize=(5.473, 2.7), dpi=180)
    plt.plot(history["loss"], label="train", color="tab:blue")
    plt.plot(history["val_loss"], label="valid", color=MAGENTA)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(loc="upper right")
    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return plt


# In[ ]:


if TEST_MODE:
    HistoryDummy = type("History", (object,), {})
    history = HistoryDummy()
    history.history = {}
    history.history["loss"] = [0.6, 0.4, 0.3, 0.2, 0.21, 0.15]
    history.history["val_loss"] = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3]
    utils_plot_training_loss(history.history)


# In[ ]:


# randomized_data_idx = list(range(len(r)))
# random.shuffle(randomized_data_idx)
# split_idx = 2 * (len(randomized_data_idx)//3) + 1
# train_set = randomized_data_idx[: split_idx]
# test_set = randomized_data_idx[split_idx: ]
# print(f"train_set: {train_set}\ntest_set: {test_set}")


# In[ ]:


# # preparing train data
# # train_set = r
# dfList_exp1_train, dfList_exp2_train = [dfList_exp1[i] for i in train_set], [dfList_exp2[i] for i in train_set]
# print(f"len(dfList_exp1_train): {len(dfList_exp1_train)}")
# print(f"len(dfList_exp2_train): {len(dfList_exp2_train)}")
# XExpTrainDict = MakeXExpDic(dfList_exp1_train, dfList_exp2_train, window_size = 250, step = 251, numSamplePoints= 18000)
# X_exp1_train_dic, X_exp2_train_dic = XExpTrainDict["X_exp1_dic"], XExpTrainDict["X_exp2_dic"]

# # preparing test data
# dfList_exp1_test, dfList_exp2_test = [dfList_exp1[i] for i in test_set], [dfList_exp2[i] for i in test_set]
# print(f"len(dfList_exp1_test): {len(dfList_exp1_test)}")
# print(f"len(dfList_exp2_test): {len(dfList_exp2_test)}")
# XExpTestDict = MakeXExpDic(dfList_exp1_test, dfList_exp2_test, window_size = 250, step = 251, numSamplePoints= 18000)
# X_exp1_test_dic, X_exp2_test_dic = XExpTestDict["X_exp1_dic"], XExpTestDict["X_exp2_dic"]


# **use the following to write tests for distro functions**

# In[ ]:


# source: https://zhiyzuo.github.io/Plot-Lorenz/
#  0 representing perfect equality, and 1 absolute inequality.
def gini(arr):
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

def lorenz_curve(X):
    ## first sort
    X = X.copy()
    X.sort()
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    fig, ax = plt.subplots(figsize=[6,6])
    ## scatter plot of Lorenz curve
    ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, 
               marker='x', color='darkgreen', s=100)
    ## line plot of equality
    ax.plot([0,1], [0,1], color='k')
    ax.set_xlabel('% of Population')
    ax.set_ylabel('% of Errors')


# In[ ]:


# window = 1000
# window_farDistro_array = DistroFAR(dic["window_farDistro_dict"][window], threshold = dic["window_EER_threshold_dict"][window])
# window_frrDistro_array = DistroFRR(dic["window_frrDistro_dict"][window], threshold = dic["window_EER_threshold_dict"][window])
# X = np.array(window_frrDistro_array)
# # print(X)
# print(gini(X))
# lorenz_curve(X)


# In[ ]:


# X = np.array(window_farDistro_array)
# # print(X)
# print(gini(X))
# lorenz_curve(X)


# In[ ]:


# keys_list = r
# values_list = window_frrDistro_array
# #Get pairs of elements


# zip_iterator = zip(keys_list, values_list)

# Distro_dict = dict(zip_iterator)

# Distro_dict = {k: v for k, v in sorted(Distro_dict.items(), key=lambda item: item[1])}

# fig, ax =plt.subplots(1,1, figsize=(8,8))

# ax.set_title('FRR Distrobution')
# print(Distro_dict)
# data = {"User": list(Distro_dict.keys()), "False Rejects": list(Distro_dict.values())}
# g = sns.barplot(x=data["User"], y=data["False Rejects"], order=data["User"],ax = ax)
# #y=EER_dict.values()

# # y_ticks = np.arange(0, .25 + 0.001, .05)

# # g.set_yticks(y_ticks)
# ax.axhline(np.mean(data["False Rejects"]), ls='--')
# ax.set_xlabel('User')
# ax.set_ylabel('False Rejects')

# fig.show()


# In[ ]:


# X = np.array(window_farDistro_array)
# # print(X)
# print(gini(X))
# lorenz_curve(X)


# In[ ]:


# auc(frrList, farList)


# In[ ]:


def getAUROCDist(window_frrList_dict, window_farList_dict, start_window_size=250, end_window_size=3000, increment_step=250):
    
    window_AUROC_dict = {}
    lst = np.arange(start_window_size, end_window_size + 1, increment_step)
    
    for w in lst:
        
        frrList = dic["window_frrList_dict"][w]
        farList = dic["window_farList_dict"][w]
        window_AUROC_dict[w] = auc(frrList, farList)
        
    return window_AUROC_dict


# In[ ]:


# AUROC_dict = getAUROCDist(dic["window_frrList_dict"], dic["window_farList_dict"])


# In[ ]:


# data = pd.read_csv('../input/wearable-assisted-ca/user10_1.docx', error_bad_lines = False, header=None, dtype = str)


# In[ ]:


# def extractTextFromDocx(path):
#     try:
#         doc = docx.Document(path)  # Creating word reader object.
#         data = ""
#         fullText = []
#         for para in doc.paragraphs:
#             fullText.append(para.text)
#             data = '\n'.join(fullText)

#     except IOError:
#         print('There was an error opening the file!')
#         return
#     return data

# # %% [code] {"execution":{"iopub.status.busy":"2022-03-18T18:34:19.771958Z","iopub.execute_input":"2022-03-18T18:34:19.772266Z","iopub.status.idle":"2022-03-18T18:34:19.793851Z","shell.execute_reply.started":"2022-03-18T18:34:19.772229Z","shell.execute_reply":"2022-03-18T18:34:19.792569Z"},"jupyter":{"outputs_hidden":false}}
# def numberOfWords(text):
#     return len(text.strip().split())

# def numberOfChars(text):
#     return len(text)

# def wordsPerMinute(text, mins):
#     return numberOfWords(text)/mins

# def charsPerMinute(text, mins):
#     return numberOfChars(text)/mins

# def classifyTypists(typistsSpeeds):
#     '''
#         WPM
#     Beginner	0 - 24
#     Intermediate	25 - 30
#     Average	31 - 41
#     Pro	42 - 54
#     Typemaster	55 - 79
#     Megaracer	80+
#     '''
#     exp2_typingspeeds = [29.96428571, 37.42857143, 44.89285714, 52.35714286, 59.82142857, 67.28571429]
#     speedDict = {"Beginner": 24, "Intermediate": 30, "Average": 41, "Pro": 54, "Typemaster": 79, "Megaracer": 1000}
    
#     keys = list(speedDict.keys())
#     for i in range(len(speedDict.keys())):
#         speedDict[keys[i]] = exp2_typingspeeds[i]
    
#     speedStats = {"Beginner": 0, "Intermediate": 0, "Average": 0, "Pro": 0, "Typemaster": 0, "Megaracer": 0}
#     typistsIDStats = {"Beginner": [], "Intermediate": [], "Average": [], "Pro": [], "Typemaster": [], "Megaracer": []}
    
#     for typist, speed in typistsSpeeds.items():
#         if speed <= speedDict["Beginner"]:
#             speedStats["Beginner"] += 1
#             typistsIDStats["Beginner"].append(typist)
            
#         elif speed <= speedDict["Intermediate"]:
#             speedStats["Intermediate"] += 1
#             typistsIDStats["Intermediate"].append(typist)
            
#         elif speed <= speedDict["Average"]:
#             speedStats["Average"] += 1
#             typistsIDStats["Average"].append(typist)
            
#         elif speed <= speedDict["Pro"]:
#             speedStats["Pro"] += 1
#             typistsIDStats["Pro"].append(typist)
            
#         elif speed <= speedDict["Typemaster"]:
#             speedStats["Typemaster"] += 1
#             typistsIDStats["Typemaster"].append(typist)
            
#         else:
#             speedStats["Megaracer"] += 1
#             typistsIDStats["Megaracer"].append(typist)
            
#     return {"speedStats": speedStats, "typistsIDStats": typistsIDStats}

# # %% [code] {"execution":{"iopub.status.busy":"2022-03-18T18:34:19.795159Z","iopub.execute_input":"2022-03-18T18:34:19.795733Z","iopub.status.idle":"2022-03-18T18:34:19.814022Z","shell.execute_reply.started":"2022-03-18T18:34:19.795693Z","shell.execute_reply":"2022-03-18T18:34:19.812934Z"},"jupyter":{"outputs_hidden":false}}
# r = [1, 2, 3, 4, 5, 6, 7, 8, 19, 21, 22, 26, 27, 28, 29, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49]
# #r = [1, 2, 3, 4, 5, 6, 7, 8, 19, 21, 22, 26, 27, 28, 29, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


# def users_typing_speed(user_ids_lst):
#     '''
#     input: 
#         user_ids_lst: list of selected user ids
#     Return {user_id: typing_speed in words per minute}
#     '''
#     typistsSpeeds = {}
#     for i in user_ids_lst:
#         user_text_data = extractTextFromDocx('../input/wearable-assisted-ca/user{0}_{1}.docx'.format(i, 2))

#         typistsSpeeds[i] = wordsPerMinute(user_text_data, 4)


# In[ ]:


# dic = classifyTypists(typistsSpeeds)
# dic


# # Divide the users using histogram

# In[ ]:


# Define a style I use a lot for boxplots:
utils_boxplot_style = dict(
    color="tab:blue",
    linewidth=0.5,
    saturation=1,
    width=0.7,
    flierprops=dict(
        marker="o", markersize=2, markerfacecolor="none", markeredgewidth=0.5
    ),
)

# Define a style I use a lot for lineplots:
utils_lineplot_style = dict(
    color="tab:blue", linewidth=0.5, marker="o", markersize=3, markeredgewidth=0.5
)


# In[ ]:


print("utility functions imported")

