import math
import numpy as np
import scipy.stats
import pickle
import os

print(os.getcwd()) # Check that your pickle file is in this folder.

with open('simulation.results', 'rb') as simfile:
    data = pickle.load(simfile)

def operation_blocking_time_from_data(data, ignore_time, end_timestamp):
    # In order to estimate blocking time from a steady state, use ignore_time
    # to ignore events before timestamp = ignore_time
    if end_timestamp < ignore_time: raise ValueError
    results = {}
    timer_totals = []
    for simulation in data:
        operation_status = list(zip(simulation['queue_timestamps']['timestamp'], 
                               simulation['queue_data']['operation'],
                               simulation['queue_data']['recovery']))
        steady_status = [event for event in operation_status if event[0] > ignore_time]
        # Figure out what the state at time ignore_time was by checking the 
        # last element to be left out of the steady_status, add that to start of 
        # events list, but with timestamp at exactly ignore_time
        state_at_ignore_time = operation_status[-len(steady_status)-1]
        steady_status = [(ignore_time, state_at_ignore_time[1], state_at_ignore_time[2])] + steady_status
        # Also we might have a case where the system ends at a blocked state, 
        # thus the timer should be left to run until that point. We handle
        # this case by adding a last_state at end_timestamp
        last_state = steady_status[-1]
        if last_state[0] > end_timestamp: raise ValueError
        end_state = (end_timestamp, last_state[1], last_state[2])
        steady_status = steady_status + [end_state]
        timer_total = 0
        
        timer_on = False 
        # Having this True should not affect the calculation, since the first event
        # will be compared against the same timestamp as itself, thus contributing
        # a net zero to timer_total
        print(steady_status)
        
        previous_timestamp = ignore_time
        for event in steady_status:
            current_timestamp = event[0]
            persons_in_operation_queue = event[1]
            persons_in_recovery_queue = event[2]
            if timer_on:
                timer_total += (current_timestamp - previous_timestamp)
            previous_timestamp = current_timestamp
            if persons_in_operation_queue and persons_in_recovery_queue: 
                timer_on = True
            else:
                timer_on = False
                
        timer_totals.append(timer_total)
        
    return timer_totals

bt_data = operation_blocking_time_from_data(data, 500, 1500)
# NB: To get blocking probability, divide mean, lower and upper by end_timestamp - ignore_time
mean = np.mean(bt_data)
lower, upper = scipy.stats.t.interval(0.95, len(bt_data)-1, 
                                            loc=mean, 
                                            scale=scipy.stats.sem(bt_data))
print("Blocking times:")
print(mean)
print(f"[{lower}, {upper}]")


    
    