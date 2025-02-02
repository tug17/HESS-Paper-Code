#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
import os
import numpy as np
import json
import pandas as pd

#############################
#         Functions
#############################
def load_metrics(path):
    with open(path, "r") as f:
        metrics = json.load(f)
    return metrics

def df2latex(df, file, bold_mask=None, fmt=None):
    print(file)
    if type(bold_mask) == type(None):
        bold_mask = np.zeros(df.shape)
    
    with open(file, "w") as f:
        f.write("index&"+"&".join(df.columns)+r"\\")
        f.write("\n")
        i = -1
        for n, row in df.iterrows():
            i += 1
            f.write(str(n))
            for j,num in enumerate(row.values):
                if bold_mask[i,j] == 1:
                    f.write(r"&\textbf{")
                else:
                    f.write(r"&")
                if type(fmt) == type(None):
                    if np.abs(num) >= 100:
                        f.write(f"{int(num): 6d}")
                    elif np.abs(num) >= 10:
                        f.write(f"{num: 6.1f}")
                    elif np.abs(num) >= 1:
                        f.write(f"{num: 6.2f}")
                    else:
                        f.write(f"{num: 6.3f}")
                elif type(fmt) == type([]):
                    if fmt[j] == 'd':
                       num = int(num)
                    f.write(format(num, fmt[j]))
                else:
                    if fmt == 'd':
                        num = int(num)
                    f.write(format(num, fmt))
                if bold_mask[i,j] == 1:
                    f.write(r"}")
                    
            f.write(r"\\")
            f.write("\n")

def get_bold_mask(df, fcn=np.argmax, n_multi_cols=3, offset=0):
    mask = np.zeros(df.shape)
    for n in range(n_multi_cols):
        if type(fcn) == type([]):
            idx = fcn[n](df.values[:, (offset+n)::n_multi_cols], axis=1)
        else:
            idx = fcn(df.values[:, (offset+n)::n_multi_cols], axis=1)
        for j in range(df.shape[0]):
            mask[j, idx[j]*n_multi_cols+n+offset] = 1
    return mask

def find_best_models(data_lstm, data_arima):
    # OBSOLETE
    all_best = []
    normalized_arima = []
    normalized_lstm = []
    normalized_hydro = []
    
    all_absolute_errors_arima = []
    all_absolute_errors_lstm = []
    

    for i in range(0,96):
        arima_q0 = data_arima['fc%s' % i].values
        lstm_q0 = data_lstm['q%s' % i].values
        measured_q0 = data_arima['obs%s' % i].values
        
        num_best_arima = 0
        num_best_lstm = 0
        num_best_hydro = 0
        
        all_abs_error = []
        all_abs_error2 = []

        for j in range(0,len(arima_q0)):
            error_arima = abs(measured_q0[j] - arima_q0[j])
            error_lstm = abs(measured_q0[j] - lstm_q0[j])
            
            if error_arima < error_lstm:
                num_best_arima += 1
            if error_lstm < error_arima:
                num_best_lstm += 1
                
            
            all_abs_error.append(error_arima)
            all_abs_error2.append(error_lstm)

        all_absolute_errors_arima.append(all_abs_error)
        all_absolute_errors_lstm.append(all_abs_error2)

        list_best = [num_best_arima, num_best_lstm]
    
        all_best.append(list_best)
    
        normalized_arima.append(num_best_arima / (num_best_arima + num_best_lstm))
        normalized_lstm.append(num_best_lstm / (num_best_arima + num_best_lstm))
        normalized_hydro.append(num_best_hydro / (num_best_arima + num_best_lstm))
    
    return all_best, np.asarray(normalized_arima), np.asarray(normalized_lstm), all_absolute_errors_arima, all_absolute_errors_lstm

def get_n_peaks(df, col_eval, n_peaks, window):
    # find the n_peaks most prominent peaks in df[col_eval] and returns peak values with the window
    peaks = []
    df_length = df.shape[0]
    for n_peak in range(n_peaks):
        i_max   = df[col_eval].argmax()
        i_start = np.max([0, i_max - window//4])
        i_end   = np.min([df_length, i_max + window])
        
        df["n_peak"] = n_peak
        peaks.append(df.iloc[i_start:i_end].copy())
        
        df = df.drop(df.index[i_start:i_end], axis=0)
    
    peaks = pd.concat(peaks, axis=0)
    return peaks

def dt(dates, format="%d/%m/%Y %H:%M"):
    if dates.tz == None:
        # make TZ aware
        return pd.to_datetime(dates, format=format).tz_localize("Europe/London").tz_convert("UTC")
    else:
        return pd.to_datetime(dates, format=format).tz_convert("UTC")

#############################
#         Classes
#############################
# class to handle plotting easier
class ModelHandler:
    def __init__(self, name, model_folder, n_trial=-1, target_name="", feat_hindcast=[], feat_forecast=[], is_external_model= False, is_final_model= False, color="r", ls="-"):
        self.name  = name
        self.color = color
        self.ls    = ls
        self.is_external_model = is_external_model
        
        if is_final_model:
            self.lg_path = model_folder
            self.hp_path = model_folder
            with open(os.path.join(self.lg_path, "features.txt"), "r") as f:
                dic = json.load(f)
            self.target_name   = dic["target_name"]
            self.feat_hindcast = dic["feat_hindcast"]
            self.feat_forecast = dic["feat_forecast"]
        else:
            self.lg_path = os.path.join(model_folder, "log", f"trial_{n_trial:02d}")
            self.hp_path = os.path.join(model_folder,  "hp", f"trial_{n_trial:02d}")
            self.target_name   = target_name
            self.feat_hindcast = feat_hindcast
            self.feat_forecast = feat_forecast