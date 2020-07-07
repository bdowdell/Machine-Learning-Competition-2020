#!/usr/bin/env python

# This program reads in pre-trained models from model_build.py
# as well as an input file to use for predicting DTC & DTS curves.
# The predicted curves are appended to the input data and written out.

# Imports
import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os

# Build the command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input well data file for making predictions")
parser.add_argument("-d", "--directory", help="Directory where fitted models are saved")
parser.add_argument("-o", "--output", help="Directory to output predicted data")
args = parser.parse_args()

# Ensure the correct number of arguments are supplied
# If not, print the help message and exit
if len(sys.argv) != 7:
    # prog_name + 6 args = 7
    parser.print_help()
    sys.exit()

# Read the inputs from the parser
input_file = args.input
model_dir = args.directory
output_dir = args.output


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


# Create a data frame to hold the input data
df = pd.read_csv(input_file)
df_iqr = iqr_filter(df.apply(np.log1p))
df_iqr = df_iqr.apply(np.expm1).interpolate(limit_area='inside')

# Get the contents of model_dir
models = [f for f in os.listdir(model_dir) if '.joblib' in f]

# Load the fitted models
# 1) Random Forest Regressor
print('Loading Random Forest Regressor joblib pickle ...')
rfr_idx = [i for i, s in enumerate(models) if 'rfr' in s]
rfr_model = joblib.load(model_dir + models[rfr_idx[0]])

# 2) Gradient Boosted Decision Tree Regressor
print('Loading Gradient Boosted Decision Tree Regressor joblib pickle ...')
gbr_idx = [i for i, s in enumerate(models) if 'gbr' in s]
gbr_model = joblib.load(model_dir + models[gbr_idx[0]])

# 3) XGBoost Regressor
print('Loading XGBoost Regressor joblib pickle ...')
xgbr_idx = [i for i, s in enumerate(models) if 'xgbr' in s]
xgbr_model = joblib.load(model_dir + models[xgbr_idx[0]])

# 4) Principal Component Regression (SVR)
print('Loading Principal Component Regression joblib pickle ...')
pcr_idx = [i for i, s in enumerate(models) if 'pcr' in s]
pcr_model = joblib.load(model_dir + models[pcr_idx[0]])

# 5) KNN Regressor
print('Loading KNN Regressor joblib pickle ...\n')
knn_idx = [i for i, s in enumerate(models) if 'knn' in s]
knn_model = joblib.load(model_dir + models[knn_idx[0]])

# Prepare data for predictions
X = df_iqr.drop(labels=['CAL', 'HRM', 'PE'], axis=1)

# Make a prediction for each model
print('Predicting y using Random Forest ...')
y_pred_rfr = rfr_model.predict(X)
print('Predicting y using Gradient Boosted Decision Tree ...')
y_pred_gbr = gbr_model.predict(X)
print('Predicting y using XGBoost Regressor ...')
y_pred_xgbr = xgbr_model.predict(X)
print('Predicting y using PCR ...')
y_pred_pcr = pcr_model.predict(X)
print('Predicting y using KNN\n')
y_pred_knn = knn_model.predict(X)

# Calculate the average ensemble prediction
print('Predicting y using average ensemble ...\n')
y_pred_ave = (y_pred_rfr + y_pred_gbr + y_pred_xgbr + y_pred_pcr + y_pred_knn) / 5

# Append the predicted DTC & DTS to df
df['DTC_pred'] = y_pred_ave[:, 0]
df['DTS_pred'] = y_pred_ave[:, 1]

# Output the data with the predicted curves
outfile = input_file.split('/')[1]  # Get the file name after the directory
outfile = outfile.split('.')[0]  # remove the file type at the end
outfile_name = output_dir + outfile + '_pred.csv'
print('Writing results to: {}'.format(outfile_name))
df.to_csv(outfile_name, index=False)
