#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sebastian Gegenleithner & Manuel Pirker
"""

#############################
#         Imports
#############################
import numpy as np
import os
import json

#############################
#         Functions
#############################
# helper functions
def get_n_peaks(df, col_eval, n_peaks, window):
    peaks = []
    df_length = df.shape[0]
    for n_peak in range(n_peaks):
        i_max   = df[col_eval].argmax()
        i_start = np.max([0, i_max - window//2])
        i_end   = np.min([df_length, i_max + window//2])
        
        peaks.append(df.iloc[i_start:i_end])
        
        df = df.drop(df.index[i_start:i_end], axis=0)
    
    return peaks

def extract_arima_metrics(path):
    with open(os.path.join(path, "model.json"),"r") as f:
        dic = json.load(f)
        
    for metric in ["NSE", "KGE", "bias"]:
        array = []
        for n_fold in range(1,6):
            array.append(dic[f"Fold_{n_fold:d}"][metric])
            
        np.savetxt(os.path.join(path, f"metric_{metric.lower()}.txt"),
                      np.array(array), delimiter=",")

    return dic

def regroup_metrics(path):
    metric = {"eval": {},
             "test": {}}
    for key in ["nse", "kge", "bias"]:
        metric["test"][key] = np.loadtxt(os.path.join(path, f"metric_{key}.txt"), delimiter=",").tolist()

    with open(os.path.join(path, "metrics.txt"), "w+") as f:
        json.dump(metric, f)  

def load_metrics(path):
    with open(path, "r") as f:
        metrics = json.load(f)
    return metrics

def find_best_model(directory, metric_key="kge"):
    valid = []
    test = []
    for path in os.listdir(directory):
        try:
            with open(os.path.join(directory, path, "metrics.txt"), "r") as f:
                metrics = json.load(f)
        except:
            break
        valid.append(np.mean(metrics["valid"][metric_key][2:]))
        test.append(np.mean(metrics["test"][metric_key]))
        
    return valid, test

# evaluate over forecasting horizont
def evaluate_multistep(obs_multistep, pred_multistep, loss_function):
    # print((obs_multistep.shape), pred_multistep.shape)
    if obs_multistep.shape[1] == pred_multistep.shape[1]:
        step_losses = [loss_function(obs_multistep[:,x,0], pred_multistep[:,x]) 
                       for x in range(pred_multistep.shape[1])] 
    else:
        step_losses = [loss_function(obs_multistep[:,0], pred_multistep[:,x]) 
                       for x in range(pred_multistep.shape[1])] 

    return step_losses

#%% metrics
def calculate_rms(observed, predicted):
    return np.sqrt(np.mean((observed - predicted)**2))

def calculate_nse(observations, predictions):
    nse = (1 - ((predictions-observations)**2).sum() / ((observations-observations.mean())**2).sum())
    return nse

def calculate_kge(observations, predictions):
    
    m1, m2 = np.nanmax((np.nanmean(observations), 1e-6)), np.nanmax((np.nanmean(predictions), 1e-6))
    r = np.sum((observations - m1) * (predictions - m2)) / (np.sqrt(np.sum((observations - m1) ** 2)) * np.sqrt(np.sum((predictions - m2) ** 2)))
    
    beta = m2 / m1
    gamma = (np.std(predictions) / m2) / (np.std(observations) / m1)
    
    # alpha = np.std(predictions) / np.std(observations)
    
    KGE =  1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    
    return KGE

def calculate_kge_var(observations, predictions):

    m1, m2 = np.nanmax((np.nanmean(observations), 1e-6)), np.nanmax((np.nanmean(predictions), 1e-6))
    gamma = (np.std(predictions) / m2) / (np.std(observations) / m1)
    
    return gamma

def calculate_kge_bias(observations, predictions):
    
    m1, m2 = np.nanmax((np.nanmean(observations), 1e-6)), np.nanmax((np.nanmean(predictions), 1e-6))
    beta = m2 / m1
    
    return beta

def calculate_kge_linear(observations, predictions):
    
    m1, m2 = np.nanmax((np.nanmean(observations), 1e-6)), np.nanmax((np.nanmean(predictions), 1e-6))
    r = np.sum((observations - m1) * (predictions - m2)) / (np.sqrt(np.sum((observations - m1) ** 2)) * np.sqrt(np.sum((predictions - m2) ** 2)))
    
    return r
    
def calculate_kge5alpha(observations, predictions):
    
    m1, m2 = np.nanmax((np.nanmean(observations), 1e-6)), np.nanmax((np.nanmean(predictions), 1e-6))
    r = np.sum((observations - m1) * (predictions - m2)) / (np.sqrt(np.sum((observations - m1) ** 2)) * np.sqrt(np.sum((predictions - m2) ** 2)))
    
    beta = m2 / m1
    gamma = (np.std(predictions) / m2) / (np.std(observations) / m1)
    
    alpha = np.std(predictions) / np.std(observations)
    
    KGE =  1 - np.sqrt((r - 1) ** 2 + (2*(alpha - 1)) ** 2 + (beta - 1) ** 2)
    
    return KGE

def calculate_bias(observations, predictions):

    numerator   = np.sum(predictions - observations)
    denominator = np.sum(observations)
    
    pbias = (numerator / denominator) * 100
    
    return pbias

def calculate_bias_fhv(observations, predictions, exceedance_prob = 0.02):
    num_total = observations.shape[0]
    num_hv    = int(num_total * exceedance_prob)

    observations_hv = np.sort(observations.flatten())[-num_hv:]
    predictions_hv  = np.sort(predictions.flatten())[-num_hv:]
    
    # sort ascending
    numerator   = np.sum(predictions_hv - observations_hv)
    denominator = np.sum(observations_hv)
    
    fhv_pbias = (numerator / denominator) * 100
    
    return fhv_pbias
    
def calculate_bias_flv(observations, predictions, exceedance_prob = 0.7):
    num_total = observations.shape[0]
    num_lv    = int(num_total * (1-exceedance_prob))

    # sort ascending
    predictions_lv  = np.sort(predictions.flatten())[:num_lv]   
    observations_lv = np.sort(observations.flatten())[:num_lv]

    # replace values close to 0 due to numerical reasons
    observations_lv[observations_lv <= 1e-6] = 1e-6
    predictions_lv[predictions_lv   <= 1e-6] = 1e-6

    # omitted to increase comparability
    # observations_lv  = np.log(observations_lv)
    # predictions_lv   = np.log(predictions_lv)
    
    obs  = np.sum(observations_lv - np.min(observations_lv))
    pred = np.sum(predictions_lv  - np.min(predictions_lv))
    
    flv_pbias = -1 * ((pred - obs) / (obs + 1e-6)) * 100
    
    return flv_pbias