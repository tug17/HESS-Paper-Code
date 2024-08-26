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

def extract_arima_metrics(path, filename = "model.json"):
    with open(os.path.join(path, filename),"r") as f:
        dic = json.load(f)
        
    for metric in ["NSE", "KGE", "bias"]:
        array = []
        for n_fold in range(1,6):
            array.append(dic[f"Fold_{n_fold:d} all"][metric])
            
        np.savetxt(os.path.join(path, f"metric_{metric.lower()}.txt"),
                      np.array(array), delimiter=",")
    return dic

def regroup_metrics(path=r"F:\11_EFFORS\python\models"):
    metric = {"eval": {},
             "test": {}}
    for key in ["nse", "kge", "bias"]:
        metric["test"][key] = np.loadtxt(os.path.join(path, f"metric_{key}.txt"), delimiter=",").tolist()

    with open(os.path.join(path, "metrics.txt"), "w+") as f:
        json.dump(metric, f)

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
            if is_external_model:
                self.target_name   = target_name
                self.feat_hindcast = feat_hindcast
                self.feat_forecast = feat_forecast
            else:
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