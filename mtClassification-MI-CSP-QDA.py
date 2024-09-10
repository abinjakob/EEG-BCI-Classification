# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:21:56 2024

Classification of the MI signal using QDA
------------------------------------------

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

#%% libraries 

import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from matplotlib import mlab
import math

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

#%% load data 

rootpath = r'L:\Cloud\NeuroCFN\Master Thesis\Classification\Data'
# EEGLab file to load (.set)
filename = 'mtP02_miblock4muband.set'
filepath = op.join(rootpath,filename)
# load file in mne 
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

#%% epoching
tmin = -0.1            
tmax = 4

# Events
event_id = {'left_imagery': 3, 'right_imagery': 6}
event_names = list(event_id.keys())

# epoching 
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['left_imagery'], event_id['right_imagery']], 
    tmin=tmin, tmax=tmax, 
    baseline= (tmin, 0), 
    preload= True,
    event_repeated = 'merge',
    reject={'eeg': 4.0}) # Reject epochs based on maximum peak-to-peak signal amplitude (PTP)


#%% prepare the data for classification 
# Imagery condition (3 & 6)
cond = ['3', '6']

# create feature vector (X)
X = epochs[cond].get_data()
# label vector (y)
y = epochs[cond].events[:,2] 

#%% 

# compute CSP on train set (using MNE csp)
ncomps = X.shape[1]
cspALL = CSP(n_components=ncomps,reg=None, log=None, transform_into = 'csp_space', norm_trace=False)
cspALL.fit(X, y)
 
# calculating the number of cols and rows for subplot
ncols = int(math.ceil(np.sqrt(ncomps)))  
nrows = int(math.ceil(ncomps / ncols))
# setting figure title
figtitle = 'Motor Execution CSP Patterns'
# creating figure
fig, ax = plt.subplots(nrows,ncols)
fig.suptitle(figtitle, fontsize=16)    
ax = ax.flatten()
for icomp in range(ncomps):
    # csp patterns 
    patterns = cspALL.patterns_[icomp].reshape(epochs.info['nchan'],-1)
    # creating a mne structure 
    evoked = mne.EvokedArray(patterns, epochs.info)
    # plotting topoplot 
    evoked.plot_topomap(times=0, axes=ax[icomp], show=False, colorbar=False)
    ax[icomp].set_title(f'Comp {icomp + 1}', fontsize=10)
# setting empty axes to false
for i in range(ncomps, len(ax)):
    ax[i].set_visible(False)  

#%% QDA classifier with 5 fold cross-validation 

# split the dataset into trainning and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -- compute CSP using mne script
ncomp = 2
csp = CSP(n_components=ncomp, reg=None, log=None, transform_into = 'csp_space', norm_trace=False)
trainCSP = csp.fit_transform(X_train, y_train)
testCSP = csp.transform(X_test)

# using log-var of CSP weights as features
X_train = np.log(np.var(trainCSP, axis=2))
X_test = np.log(np.var(testCSP, axis=2))

# define a pipeline with preprocessing (scaling) and QDA classifier
pipeline = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())

# parameter grid for QDA
param_grid = {'quadraticdiscriminantanalysis__reg_param': [0.0, 0.1, 0.5, 1.0]}

# apply cros-validaion on training set to find best SVM parameters
clf = GridSearchCV(pipeline, param_grid, cv=5)
# train the pipeline
clf.fit(X_train, y_train)

# display best parameters found by GridSearchCV
print(f'Best Parameters Found: {clf.best_params_}')

# make predictions
y_pred = clf.predict(X_test)

# generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# calculate model performance
# accuracy
accuracy = accuracy_score(y_test, y_pred)
# precision (positive predictive value)
precision = precision_score(y_test, y_pred, labels=[cond[0],cond[1]], average= 'weighted')
# recall (sensitivy or true positive rate)
recall = recall_score(y_test, y_pred, labels=[cond[0],cond[1]], average= 'weighted')
# f1 score (equillibrium between precision and recall)
f1score = f1_score(y_test, y_pred, labels=[cond[0],cond[1]], average= 'weighted')

# print model performance 
print('Confusion Matrix')
print(cm)
print('Model Performance Metrics')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1score*100:.2f}%') 

