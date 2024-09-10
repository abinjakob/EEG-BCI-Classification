# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:14:23 2024

Model Training for CCA-SVM Model for SSVEP Classification 
---------------------------------------------------------
Feature used: CCA Correlation Values for stimulus frequencies and its harmonics
Classification: SVM classifier 
                - scaling using StandarScalar
                - hyperparameter tuning using GridSearchCV

@author: Abin Jacob
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
"""

# libraries 
import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from joblib import dump

from sklearn.metrics import confusion_matrix, accuracy_score, PrecisionRecallDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA


# -- prameters 
# epoching 
tmin = -0.2
tmax = 4
# events
event_id = {'stim_L15': 10, 'stim_L20': 11, 'stim_R15': 12, 'stim_R20': 13}
event_names = list(event_id.keys()) 
# files 
rootpath = r'L:\Cloud\NeuroCFN\Master Thesis\Classification\Data'
# EEGLab file to load (.set)
filename  = 'mtP02_ssvepblock1.set'
modelname = 'mtP02_SSVEPblock1model'

# load data 
print('\nloading data...........')   
filepath = op.join(rootpath,filename)
raw = mne.io.read_raw_eeglab(filepath, eog= 'auto', preload= True)

# extracting events 
events, eventinfo = mne.events_from_annotations(raw, verbose= False)

# epoching 
epochs = mne.Epochs(
    raw, 
    events= events, 
    event_id= [event_id['stim_L15'], event_id['stim_L20'], event_id['stim_R15'], event_id['stim_R20']], 
    tmin=tmin, tmax=tmax, 
    baseline= None, 
    preload= True,
    event_repeated = 'merge',
    reject={'eeg': 3.0}) 


# -- computing CCA 
print('\ncomputing CCA...........')  
# number of epochs and samples 
numEpochs, _, tpts = epochs.get_data().shape
# eeg data from the epocs 
eegEpoch = epochs.get_data()
# stimulation frequencies
freqs = [15, 20]
# sampling frequency
fs = epochs.info["sfreq"]
# duration of epochs 
duration = tpts/fs
# generating time vector
t = np.linspace(0, duration, tpts, endpoint= False)

# initialising array to store features
CCAfeatures = []

# loop over epochs 
for iEpoch in range(numEpochs):
    # extract the X array
    X_data = eegEpoch[iEpoch,:,:].T
    # initialise array to store featues for each epoch
    epochFeat = []
    # loop over frequencies
    for i, iFreq in enumerate(freqs):    
        # create the sine and cosine signals for 1st harmonics
        sine1 = np.sin(2 * np.pi * iFreq * t)
        cos1 = np.cos(2 * np.pi * iFreq * t)
        # create the sine and cosine signals for 2nd harmonics
        sine2 = np.sin(2 * np.pi * (2 * iFreq) * t)
        cos2 = np.cos(2 * np.pi * (2 * iFreq) * t)        
        # create Y vector 
        Y_data = np.column_stack((sine1, cos1, sine2, cos2))       
        # performing CCA
        # considering the first canonical variables
        cca = CCA(n_components= 1)
        # compute cannonical variables
        cca.fit(X_data, Y_data)
        # return canonical variables
        Xc, Yc = cca.transform(X_data, Y_data)
        corr = np.corrcoef(Xc.T, Yc.T)[0,1]       
        # store corr values for current epoch
        epochFeat.append(corr)  
    # store features
    CCAfeatures.extend(epochFeat)


# -- preparing data for training 
# create labels 
labels = epochs.events[:,2]
for i in range(0,len(labels)):
    if labels[i]==10 or labels[i]==12:
        labels[i] = 15
    else:
        labels[i] = 20

# feature vector (X)
X = np.array(CCAfeatures).reshape(numEpochs, -1)
# label vector (y)
y = labels 


# -- SVM model training
print('\ntraining SVM model...........')
# define a pipeline with preprocessing (scaling) and SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC())

# parameter grid for SVM
param_grid = {
    'svc__C': [0.1, 1, 10, 100],  # SVM regularization parameter
    'svc__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
}

# apply cros-validaion on training set to find best SVM parameters
clf = GridSearchCV(pipeline, param_grid, cv=5)
# train the pipeline
clf.fit(X, y)

# display best parameters found by GridSearchCV
print(f'Best Parameters Found: {clf.best_params_}')


# -- save the model
print('\nsaving SVM model...........')  
modeldir = op.join(rootpath,modelname)
# dumping to disk
dump(clf, modeldir)     


