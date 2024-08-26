#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Manuel Pirker
"""

#############################
#         Imports
#############################
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import os
import sys

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(17)

#############################
#         Functions
#############################
@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stderr = old_stderr


#############################
#         Classes
#############################
#%% Hindcast LSTM
class Hindcast:
    def __init__(self, random_seed=17):
        tf.keras.utils.set_random_seed(random_seed)
        self.name = "HLSTM"

    def build_model(hp):
        ## input for hincast and forecast model
        inp_hincast  = tf.keras.layers.Input(shape=(hp["hindcast_len"], hp["n_features_hc"]))
        inp_forecast = tf.keras.layers.Input(shape=(hp["forecast_len"], hp["n_features_fc"]))
        
        ## hincast model
        hidden  = inp_hincast
        
        # add LSTM layer
        lstm_hincast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                      activation   = 'tanh', 
                                      dropout      = 0, #hp["lstm_dropout"],
                                      return_state = True,
                                      # return_sequences = True,
                                      )(hidden)

        ## Pass the last hidden state and cell state to another LSTM layer
        _, last_hidden_state, last_cell_state = lstm_hincast
        
        last_hidden_state = tf.keras.layers.Dense(hp["lstm_unit"], activation ='linear')(last_hidden_state)
        last_cell_state   = tf.keras.layers.Dense(hp["lstm_unit"], activation ='linear')(last_cell_state)
        
        ## forecast model
        hidden = inp_forecast
        
        # add lstm and set inital state
        lstm_forecast = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                             activation   = 'tanh', 
                                             dropout      = 0, #hp["lstm_dropout"],
                                             return_sequences=True,
                                             #go_backwards=True,
                                            )(hidden,
                                              initial_state=[last_hidden_state, last_cell_state],
                                             )
        # flatten layer
        hidden = tf.keras.layers.Flatten()(lstm_forecast)
        
        hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # fully connected
        hidden = tf.keras.layers.Dense(hp["target_len"], activation='relu')(hidden)
        
        ## create model                               
        model = tf.keras.Model(inputs=[inp_hincast, inp_forecast], outputs=hidden)

        #model.summary()
        return model

#%% eLSTM
class eLSTM:
    def __init__(self, random_seed=17):
        tf.keras.utils.set_random_seed(random_seed)
        self.name = "eLSTM"

    def build_model(hp):
        ## input
        inp_hincast  = tf.keras.layers.Input(shape=(hp["hindcast_len"], hp["n_features_hc"])) # ignored for eLSTM
        inp_forecast = tf.keras.layers.Input(shape=(hp["forecast_len"], hp["n_features_fc"]))
        
        # add LSTM layer
        hidden = tf.keras.layers.LSTM(units  = hp["lstm_unit"],
                                      activation   = 'tanh', 
                                      dropout      = 0, #hp["lstm_dropout"],
                                      return_state = False,
                                      return_sequences = True,
                                      )(inp_forecast)

        # flatten layer
        hidden = tf.keras.layers.Flatten()(hidden)
        
        hidden = tf.keras.layers.Dropout(hp["dropout_rate"])(hidden)
        
        # fully connected
        hidden = tf.keras.layers.Dense(hp["target_len"], activation='relu')(hidden)
        
        ## create model                               
        model = tf.keras.Model(inputs=[inp_hincast, inp_forecast], outputs=hidden)

        #model.summary()
        return model
