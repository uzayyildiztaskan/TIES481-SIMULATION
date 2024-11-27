
import pickle
import os

print(os.getcwd())

with open('simulation.results', 'rb') as simfile:
    data = pickle.load(simfile)

def blocking_time_from_data(data):
    timer_totals = []
    for simulation in data:
        print(simulation)
        operation_status = list(zip(simulation['queue_timestamps']['timestamp'], 
                               simulation['queue_data']['operation']))
        
        timer_total = 0
        timer_on = False
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
    
print(blocking_time_from_data(data))

    
    