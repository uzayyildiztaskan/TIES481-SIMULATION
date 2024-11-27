import math
import numpy as np
import scipy.stats
import pickle
import os

print(os.getcwd())

with open('simulation.results', 'rb') as simfile:
    data = pickle.load(simfile)

def blocking_time_from_data(data):
    
    results = {}
    timer_totals = []
    for simulation in data:
        operation_status = list(zip(simulation['queue_timestamps']['timestamp'], 
                               simulation['queue_data']['operation']))
        
        timer_total = 0
        timer_on = False # We assume an empty hospital in the beginning
        previous_timestamp = 0
        for event in operation_status:
            current_timestamp = event[0]
            persons_in_queue = event[1]
            if timer_on:
                timer_total += (current_timestamp - previous_timestamp)
            previous_timestamp = current_timestamp
            if persons_in_queue: 
                timer_on = True
            else:
                timer_on = False
                
        timer_totals.append(timer_total)
        
    return timer_totals


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

bt_data = blocking_time_from_data(data)
mean_confidence_interval(bt_data)
mean, lower, upper = mean_confidence_interval(bt_data)
print(mean)
print(f"{lower}, {upper}")


    
    