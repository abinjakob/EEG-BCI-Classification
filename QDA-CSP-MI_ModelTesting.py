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
from joblib import load

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


# -- QDA model classification
print(f'\nLoading {cspname} CSP weights...........')
cspimport = op.join(rootpath,cspname)
csp = load(cspimport)

print(f'\nLoading {modelname} SVM model...........')
modelimport = op.join(rootpath,modelname)
clf = load(modelimport)

# appying csp weights 
testCSP = csp.transform(X)

# using log-var of CSP weights as features
X_test = np.log(np.var(testCSP, axis=2))


# make predictions
y_pred = clf.predict(X_test)

# generate the confusion matrix
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

# calculate model performance
# accuracy
accuracy = accuracy_score(y, y_pred)
# precision (positive predictive value)
precision = precision_score(y, y_pred, labels=[cond[0],cond[1]], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y, y_pred, labels=[cond[0],cond[1]], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y, y_pred, labels=[cond[0],cond[1]], average= 'weighted')

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%') 


