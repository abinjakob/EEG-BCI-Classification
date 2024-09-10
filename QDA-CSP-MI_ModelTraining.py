# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:27:25 2024

The script is used for the offline classification of the MI EEG data.  

Feature used: log-var of CSP spatial filtred data

Classification: LDA classifier with 5-Fold crossvalidation
                - spliting data using train_test_split
                - scaling using StandarScalar
                - hyperparameter tuning using GridSearchCV


@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
      
"""

# -- libraries 

import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import math
from joblib import dump

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP


# -- params
# epoching
tmin = -0.1            
tmax = 4
# events
event_id = {'left_imagery': 3, 'right_imagery': 6}
event_names = list(event_id.keys())
# files
rootpath  = r'L:\Cloud\NeuroCFN\Master Thesis\Classification\Data'
filename  = 'mtP02_miblock1muband.set'
modelname = 'mtP02_MIblock1model'
cspname   = 'mtP02_MIblock1cspweights'


# load data 
print('\nloading data...........')  
filepath = op.join(rootpath,filename)
raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)

# eeg paramters 
sfreq = raw.info['sfreq']
# eeg signal
EEG = raw.get_data()
nchannels, nsamples = EEG.shape
# channel names 
chnames = raw.info['ch_names'] 
# extracting events 
events, eventinfo = mne.events_from_annotations(raw, verbose= False)

# epoching 
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['left_imagery'], event_id['right_imagery']], 
    tmin=tmin, tmax=tmax, 
    baseline= (tmin, 0), 
    preload= True,
    event_repeated = 'merge',
    reject={'eeg': 4.0}) 


# -- preparing data for training 
# Imagery condition (3 & 6)
cond = ['3', '6']
# create feature vector (X)
X = epochs[cond].get_data()
# label vector (y)
y = epochs[cond].events[:,2] 


# -- QDA model training
# computing CSP
print('\ncomputing csp...........')  
ncomp = 2
csp = CSP(n_components=ncomp, reg=None, log=None, transform_into = 'csp_space', norm_trace=False)
trainCSP = csp.fit_transform(X, y)
# using log-var of CSP weights as features
X_train = np.log(np.var(trainCSP, axis=2))

print('\ntraining QDA algorithm...........')  
# define a pipeline with preprocessing (scaling) and QDA classifier
pipeline = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())
# parameter grid for QDA
param_grid = {'quadraticdiscriminantanalysis__reg_param': [0.0, 0.1, 0.5, 1.0]}
# apply cros-validaion on training set to find best SVM parameters
clf = GridSearchCV(pipeline, param_grid, cv=5)
# train the pipeline
clf.fit(X_train, y)

# display best parameters found by GridSearchCV
print(f'Best Parameters Found: {clf.best_params_}')


# -- saving CSP weights and QDA model
print('\nsaving...........') 
# saving csp weights
cspdir = op.join(rootpath,cspname)
dump(csp, cspdir) 
# saving model
modeldir = op.join(rootpath,modelname)
dump(clf, modeldir) 

