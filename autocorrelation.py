
import math
import numpy as np
import scipy.stats
import pickle
import os

print(os.getcwd()) # Check that your pickle file is in this folder.

# TODO: Currently parameters such as ignore_time are hard-coded.
# This would need them to be added into the passable parameters.

#%%
with open('multi_config_simulation.pkl', 'rb') as simfile:
    data = pickle.load(simfile)

#%%

patient_data = []

# After some testing, 4p5r was chosen as the configuration of choice.
# No urgent patients, exponential distributions for all time variables.
sim_type = '3p3r'

for i in range(len(data[sim_type])):
    run = data[sim_type][i]['summary_statistics']['full_patient_records']
    
    ready_patients = [patient for patient in run if patient['current_stage'] == 'DISCHARGED' 
                      and patient['arrival_time'] > ignore_time and not patient['urgent']]
    patient_waits = [patient['prep_wait_time'] + patient['operation_wait_time'] for patient in ready_patients]
    patient_timestamps = [patient['arrival_time'] for patient in ready_patients]
    
    patients = list(zip(patient_timestamps, patient_waits))
    patients = sorted(patients, key = lambda x: x[0])
    patient_data.append(patients)
    
    
#%%
auto_correlations = []
for run in patient_data[0:10]: #Could use all 20 here, but assignment said 10
    
    xp = [p[0] for p in run]
    fp = [p[1] for p in run]
    
    # Transform disparate data points to an even sampling via linear interpolation   
    sampling_points = np.linspace(550, 1450, num=10)
    samples = np.interp(sampling_points, xp, fp)
    
    mean = sum(samples) / len(samples) 
    var = sum([(x - mean)**2 for x in samples]) / len(samples) 
    
    # Normalize data
    nsamples = [x - mean for x in samples]
    
    lag = 1
    c = sum([nsamples[lag:][i] * nsamples[:-lag][i] for i in range(len(samples) - lag)]) / len(samples) / var
    
    auto_correlations.append(c) 

# Fisher z-transform in order to properly average over correlations,
# see eg. https://blogs.sas.com/content/iml/2017/09/20/fishers-transformation-correlation.html

zs = [np.arctanh(c) for c in auto_correlations]
z_mean = np.mean(zs)
r_mean = np.tanh(z_mean)

print(f"Observed average autocorrelation over lag 1: {r_mean}")
if r_mean > 1: print("Error, correlation impossible to exceed 1")
elif r_mean > 0.7: print("Strong positive correlation")
elif r_mean > 0.5: print("Moderate positive correlation")
elif r_mean > 0.3: print("Low positive correlation")
elif r_mean > 0: print("Insignificant positive correlation")
elif r_mean > -0.3: print("Insignificant negative correlation")
elif r_mean > -0.5: print("Low negative correlation")
elif r_mean > -0.7: print("Moderate negative correlation")
elif r_mean > -1: print("Strong negative correlation")
else: print("Error, correlation impossible to be below -1")