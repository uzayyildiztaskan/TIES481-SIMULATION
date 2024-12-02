import math
import numpy as np
import scipy.stats
import pickle
import os
from config import SimulationConfig

print(os.getcwd()) # Check that your pickle file is in this folder.
ignore_time = SimulationConfig.IGNORE_TIME
end_timestamp = SimulationConfig.SIMULATION_TIME + SimulationConfig.IGNORE_TIME
is_paired = SimulationConfig.IS_PAIRED

with open('simulation.results', 'rb') as simfile:
    data = pickle.load(simfile)


def operation_blocking_time_from_data(data, ignore_time, end_timestamp):
    # In order to estimate from a steady state, use ignore_time
    # to ignore events before timestamp = ignore_time
    if end_timestamp < ignore_time: raise ValueError

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

def preparation_capacities(data):
    return [simulation['summary_statistics']['resource_utilization']['prep_rooms']['mean'] for simulation in data]
    
def average_queue_lengths_from_data(data, location, ignore_time, end_timestamp):
    # In order to estimate from a steady state, use ignore_time
    # to ignore events before timestamp = ignore_time
    if end_timestamp < ignore_time: raise ValueError
    queue_averages = []
    for simulation in data:
        queue_status = list(zip(simulation['queue_timestamps']['timestamp'], simulation['queue_data'][location]))
        steady_status = [event for event in queue_status if event[0] > ignore_time]
        # Figure out what the state at time ignore_time was by checking the 
        # last element to be left out of the steady_status, add that to start of 
        # events list, but with timestamp at exactly ignore_time
        state_at_ignore_time = queue_status[-len(steady_status)-1]
        steady_status = [(ignore_time, state_at_ignore_time[1])] + steady_status
        # Also we might have a case where the system ends at a blocked state, 
        # thus the timer should be left to run until that point. We handle
        # this case by adding a last_state at end_timestamp
        last_state = steady_status[-1]
        if last_state[0] > end_timestamp: raise ValueError
        end_state = (end_timestamp, last_state[1])
        steady_status = steady_status + [end_state]
        
        previous_timestamp = ignore_time 
        queue_mass = 0
        # This in effect integration for a step function
        
        for event in steady_status:
            current_timestamp = event[0]
            queue_mass += event[1] * (current_timestamp - previous_timestamp)
            previous_timestamp = current_timestamp
        
        queue_averages.append(queue_mass / (end_timestamp - ignore_time))
   
    return queue_averages
    
# Perform pairwise comparisons
def pairwise_comparisons(metric_extractor, config1, config2):
    data1 = metric_extractor(config1)
    data2 = metric_extractor(config2)
    differences = np.array(data1) - np.array(data2)
    mean_diff, lower, upper = confidence_interval(differences)
    t_stat, p_val = scipy.stats.ttest_rel(data1, data2)
    return mean_diff, lower, upper, t_stat, p_val



# Perform statistical tests
def comparative_analysis(data, is_paired):
    # Handle single configuration vs multiple configurations
    if isinstance(data, list):
        data = {"default": data}  # Wrap single configuration in a dict for uniform processing
    configurations = list(data.keys())
    metrics = {
        "Operation Blocking Time": operation_blocking_time_from_data,
        "Preparation Queue Length": lambda d: average_queue_lengths_from_data(d, "preparation", ignore_time, end_timestamp),
        "Preparation Utilization": preparation_capacities
    }

    for metric_name, metric_func in metrics.items():
        print(f"\n=== Analysis for {metric_name} ===")
        for i in range(len(configurations)):
            for j in range(i + 1, len(configurations)):
                config1, config2 = configurations[i], configurations[j]
                if is_paired:
                    result = pairwise_comparisons(metric_func, data[config1], data[config2])
                    mean_diff, lower, upper, t_stat, p_val = result
                    print(f"{config1} vs {config2} (Paired):")
                    print(f"  Mean Difference: {mean_diff:.4f}")
                    print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
                    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
                else:
                    data1, data2 = metric_func(data[config1]), metric_func(data[config2])
                    t_stat, p_val = scipy.stats.ttest_ind(data1, data2)
                    mean1, lower1, upper1 = confidence_interval(data1)
                    mean2, lower2, upper2 = confidence_interval(data2)
                    print(f"{config1} vs {config2} (Independent):")
                    print(f"  {config1} Mean: {mean1:.4f}, 95% CI: [{lower1:.4f}, {upper1:.4f}]")
                    print(f"  {config2} Mean: {mean2:.4f}, 95% CI: [{lower2:.4f}, {upper2:.4f}]")
                    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")


queue_data = average_queue_lengths_from_data(data, "preparation", ignore_time, end_timestamp)
mean = np.mean(queue_data)
lower, upper = scipy.stats.t.interval(0.95, len(queue_data)-1, 
                                            loc=mean, 
                                            scale=scipy.stats.sem(queue_data))

print(f"Preparation queue length average: {mean}")
print(f"95% confidence interval [{lower}, {upper}]")

bt_data = operation_blocking_time_from_data(data, ignore_time, end_timestamp)

# NB: To get blocking probability, divide mean, lower and upper by (end_timestamp - ignore_time)
mean = np.mean(bt_data)
lower, upper = scipy.stats.t.interval(0.95, len(bt_data)-1, 
                                            loc=mean, 
                                            scale=scipy.stats.sem(bt_data))

print(f"Blocking time average: {mean}")
print(f"95% confidence interval [{lower}, {upper}]")

cap_data = preparation_capacities(data)
mean = np.mean(cap_data)
lower, upper = scipy.stats.t.interval(0.95, len(cap_data)-1, 
                                            loc=mean, 
                                            scale=scipy.stats.sem(cap_data))

print(f"Average utilization: {mean}")
print(f"95% confidence interval [{lower}, {upper}]")

#comparative_analysis(data, is_paired)
    
    