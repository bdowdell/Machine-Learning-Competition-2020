#!/usr/bin/env python

# This program implements the notebook solution 
# "datadrivenpancakes_solution_submission_3.ipynb"
# Some steps taken in the notebook will be shortcut to save time
# The user will pass the input file for train/test as a command line argument
# The program will perform train & test routines
# The fitted classifier(s) will be output using pickle for model persistance
# The user can use the pickled estimator in a second program to 
# perform estimation on new data

# Standard Imports
import pandas as pd
import numpy as np
import time
import joblib
import argparse
import sys

# SciKit-Learn imports
# Pre-processing & Model Selection Utilities
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
# Regression Models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# KNN
from sklearn.neighbors import KNeighborsRegressor
# Dimensionality Reduction
from sklearn.decomposition import PCA
# Evaluation Metrics
from sklearn.metrics import mean_squared_error
# Pipeline
from sklearn.pipeline import make_pipeline
# Multi-output regressor
from sklearn.multioutput import MultiOutputRegressor
# XGBoost
import xgboost as xgb

# Set the random state
random_state = 42

# Build the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="input file (.csv) to read")
parser.add_argument("-o", "--output_dir", help="Output directory for pickles")
args = parser.parse_args()

if len(sys.argv) != 5:
    parser.print_help()
    sys.exit()

# Get the input file name and the output directory name
input_file = args.input
output_dir = args.output_dir

# Create an output file to save program log
log_file = output_dir + 'log_file.txt'
f = open(log_file, 'w')

# Write initial data to log_file
print('model_build.py launch time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('model_build.py launch time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
print('\nInput file for model training: {}'.format(input_file))
f.write('\nInput file for model training: {}\n'.format(input_file))


# Helper functions
def iqr_filter(data):
    """
    Takes a DataFrame containing well log data and filters outliers using Interquartile Range
    
    Parameters:
    data, pandas.DataFrame
        The input data to be filtered.  Each column should be a well log curve

    Returns:
    df_filt, pandas.DataFrame
        The filtered well curves
    """
    # create a copy of the original data
    df_copy = data.copy()
    # create a data frame containing the cleaned version of the data using IQR
    df_filt = pd.DataFrame(index=df_copy.index)
    # filter each column
    for col in df_copy.columns.tolist():
        q1 = df_copy[col].quantile(0.25)
        q3 = df_copy[col].quantile(0.75)
        IQR = q3 - q1
        iqr_filt = (df_copy[col] >= q1 - 1.5 * IQR) & (df_copy[col] <= q3 + 1.5 * IQR)
        df_filt[col] = df_copy[col].loc[iqr_filt]
    return df_filt


# Define custom scorer
def rmse(y_real, y_pred):
    return np.sqrt(mean_squared_error(y_real, y_pred))


# Create a data frame to hold the input file
df_in = pd.read_csv(input_file)

# Replace -999.0 values with NaN
df_in.replace(to_replace=-999.0, value=np.nan, inplace=True)

# Set value limits
df_in.loc[df_in['CNC'] < 0.0, ['CNC']] = np.nan
df_in.loc[df_in['CNC'] > 1.0, ['CNC']] = np.nan
df_in.loc[df_in['GR'] < 0.0, ['GR']] = np.nan
df_in.loc[df_in['GR'] > 300.0, ['GR']] = np.nan
df_in.loc[df_in['PE'] < 0.0, ['PE']] = np.nan
df_in.loc[df_in['ZDEN'] < 0.0, ['ZDEN']] = np.nan

# Use Inter-Quartile Range Filtering to remove statistical outliers
df_clean = iqr_filter(df_in.apply(np.log1p))
df_clean = df_clean.apply(np.expm1)

# Replace filtered version of DTC & DTS with originals
df_clean['DTC'] = df_in['DTC']
df_clean['DTS'] = df_in['DTS']

# Prepare for Train-Test Split
df_working = df_clean.copy()
df_working.drop(labels=['CAL', 'HRM', 'PE'], axis=1, inplace=True)
df_working.loc[:, ['CNC', 'GR', 'HRD', 'ZDEN']].interpolate(limit_area='inside', inplace=True)
df_working.dropna(inplace=True)
df_working.reset_index(drop=True, inplace=True)

# Create X & y variables
X = df_working.drop(labels=['DTC', 'DTS'], axis=1)
y = df_working[['DTC', 'DTS']]

# Split the data into Train & Test set (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# The solution uses an average ensemble of multiple different estimators
# Build a pipeline for each estimator step
# The overall pipeline looks like:
#       a) Log-Transform
#       b) StandardScaler()
#       c) PCA decomposition keeping the first 3 components
#       d) Estimator w/ best parameters previously defined using GridSearchCV (see notebook)

print('\nBeginning model fitting ...\n')
f.write('\nBeginning model fitting ...\n')

# 1) Random Forest
# Build the pipeline
pipe_rfr_pca = make_pipeline(
    FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
    StandardScaler(),
    PCA(n_components=3, random_state=random_state),
    RandomForestRegressor(
        max_depth=20,
        max_features=2,
        n_estimators=2000,
        random_state=random_state,
        n_jobs=-1
    )
)
# Fit the Random Forest Pipeline
print('Random Forest Pipeline Fit start time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('Random Forest Pipeline Fit start time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
pipe_rfr_pca.fit(X_train, y_train)
print('Random Forest Pipeline Fit end time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('Random Forest Pipeline Fit end time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
# Print metrics
print('Random Forest Train RMSE: {:.4f}'.format(rmse(y_train, pipe_rfr_pca.predict(X_train))))
f.write('Random Forest Train RMSE: {:.4f}\n'.format(rmse(y_train, pipe_rfr_pca.predict(X_train))))
print('Random Forest Test RMSE: {:.4f}'.format(rmse(y_test, pipe_rfr_pca.predict(X_test))))
f.write('Random Forest Test RMSE: {:.4f}\n'.format(rmse(y_test, pipe_rfr_pca.predict(X_test))))
# Output the fitted RFR model
print('Saving fitted model ...')
f.write('Saving fitted model ...\n')
rfr_pickle = output_dir + 'rfr_fitted.joblib'
joblib.dump(pipe_rfr_pca, rfr_pickle)
print('Fitted model saved: {}\n\n'.format(rfr_pickle))
f.write('Fitted model saved: {}\n\n'.format(rfr_pickle))

# 2) Gradient Boosted Decision Trees
# Build the pipeline
pipe_gbr_pca = make_pipeline(
    FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
    StandardScaler(),
    PCA(n_components=3, random_state=random_state),
    MultiOutputRegressor(GradientBoostingRegressor(
        learning_rate=0.01,
        max_depth=9,
        max_features=2,
        n_estimators=500,
        n_iter_no_change=5,
        subsample=0.8,
        random_state=random_state
    ), n_jobs=-1)
)
# Fit the GBR Pipeline
print('Gradient Boosted Decision Trees Pipeline Fit start time: {}'.format(
    time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('Gradient Boosted Decision Trees Pipeline Fit start time: {}\n'.format(
    time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
pipe_gbr_pca.fit(X_train, y_train)
print('Gradient Boosted Decision Trees Pipeline Fit end time: {}'.format(
    time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('Gradient Boosted Decision Trees Pipeline Fit end time: {}\n'.format(
    time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
# Print metrics
print('Gradient Boosted Train RMSE: {:.4f}'.format(rmse(y_train, pipe_gbr_pca.predict(X_train))))
f.write('Gradient Boosted Train RMSE: {:.4f}\n'.format(rmse(y_train, pipe_gbr_pca.predict(X_train))))
print('Gradient Boosted Test RMSE: {:.4f}'.format(rmse(y_test, pipe_gbr_pca.predict(X_test))))
f.write('Gradient Boosted Test RMSE: {:.4f}\n'.format(rmse(y_test, pipe_gbr_pca.predict(X_test))))
# Output the fitted GBR model
print('Saving fitted model ...')
f.write('Saving fitted model ...\n')
gbr_pickle = output_dir + 'gbr_fitted.joblib'
joblib.dump(pipe_gbr_pca, gbr_pickle)
print('Fitted model saved: {}\n\n'.format(gbr_pickle))
f.write('Fitted model saved: {}\n\n'.format(gbr_pickle))

# 3) XGBoost
# Build the pipeline
pipe_xgbr_pca = make_pipeline(
    FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
    StandardScaler(),
    PCA(n_components=3, random_state=random_state),
    MultiOutputRegressor(
        xgb.XGBRegressor(
            colsample_bytree=1,
            learning_rate=0.01,
            max_depth=9,
            n_estimators=1000,
            subsample=0.8,
            random_state=random_state
        ),
        n_jobs=-1
    )
)
# Fit the XGBR pipeline
print('XGBoost Pipeline Fit start time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('XGBoost Pipeline Fit start time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
pipe_xgbr_pca.fit(X_train, y_train)
print('XGBoost Pipeline Fit end time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('XGBoost Pipeline Fit end time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
# Print metrics
print('XGBoost Train RMSE: {:.4f}'.format(rmse(y_train, pipe_xgbr_pca.predict(X_train))))
f.write('XGBoost Train RMSE: {:.4f}\n'.format(rmse(y_train, pipe_xgbr_pca.predict(X_train))))
print('XGBoost Test RMSE: {:.4f}'.format(rmse(y_test, pipe_xgbr_pca.predict(X_test))))
f.write('XGBoost Test RMSE: {:.4f}\n'.format(rmse(y_test, pipe_xgbr_pca.predict(X_test))))
# Output the XGBR fitted model
print('Saving fitted model ...')
f.write('Saving fitted model ...\n')
xgbr_pickle = output_dir + 'xgbr_fitted.joblib'
joblib.dump(pipe_xgbr_pca, xgbr_pickle)
print('Fitted model saved: {}\n\n'.format(xgbr_pickle))
f.write('Fitted model saved: {}\n\n'.format(xgbr_pickle))

# 4) Principal Component Regression (PCR)
# Build the pipeline
pipe_pcr_svr = make_pipeline(
    FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
    StandardScaler(),
    PCA(n_components=3, random_state=random_state),
    MultiOutputRegressor(
        SVR(
            gamma=1.0,
            kernel='rbf'
        ),
        n_jobs=-1
    )
)
# Fit the PCR Pipeline
print('PCR Pipeline Fit start time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('PCR Pipeline Fit start time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
pipe_pcr_svr.fit(X_train, y_train)
print('PCR Pipeline Fit end time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('PCR Pipeline Fit end time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
# Print metrics
print('PCR Train RMSE: {:.4f}'.format(rmse(y_train, pipe_pcr_svr.predict(X_train))))
f.write('PCR Train RMSE: {:.4f}\n'.format(rmse(y_train, pipe_pcr_svr.predict(X_train))))
print('PCR Test RMSE: {:.4f}'.format(rmse(y_test, pipe_pcr_svr.predict(X_test))))
f.write('PCR Test RMSE: {:.4f}\n'.format(rmse(y_test, pipe_pcr_svr.predict(X_test))))
# Output the PCR fitted model
print('Saving fitted model ...')
f.write('Saving fitted model ...\n')
pcr_pickle = output_dir + 'pcr_fitted.joblib'
joblib.dump(pipe_pcr_svr, pcr_pickle)
print('Fitted model saved: {}\n\n'.format(pcr_pickle))
f.write('Fitted model saved: {}\n\n'.format(pcr_pickle))

# 5) KNN Regressor
# Build the pipeline
pipe_knn = make_pipeline(
    FunctionTransformer(func=np.log1p, inverse_func=np.expm1),
    StandardScaler(),
    PCA(n_components=3, random_state=random_state),
    KNeighborsRegressor(n_neighbors=9, n_jobs=-1)
)
# Fit the KNN Pipeline
print('KNN Pipeline Fit start time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('KNN Pipeline Fit start time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
pipe_knn.fit(X_train, y_train)
print('KNN Pipeline Fit end time: {}'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('KNN Pipeline Fit end time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
# Print metrics
print('KNN Train RMSE: {:.4f}'.format(rmse(y_train, pipe_knn.predict(X_train))))
f.write('KNN Train RMSE: {:.4f}\n'.format(rmse(y_train, pipe_knn.predict(X_train))))
print('KNN Test RMSE: {:.4f}'.format(rmse(y_test, pipe_knn.predict(X_test))))
f.write('KNN Test RMSE: {:.4f}\n'.format(rmse(y_test, pipe_knn.predict(X_test))))
# Output the KNN fitted model
print('Saving fitted model ...')
f.write('Saving fitted model ...\n')
knn_pickle = output_dir + 'knn_fitted.joblib'
joblib.dump(pipe_knn, knn_pickle)
print('Fitted model saved: {}\n\n'.format(knn_pickle))
f.write('Fitted model saved: {}\n\n'.format(knn_pickle))

print('Complete.  Time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))
f.write('Complete.  Time: {}\n'.format(time.strftime('%d %b %Y %H:%M:%S', time.localtime())))

# Close the log file
f.close()
print('Log file saved: {}'.format(log_file))
