import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from monitoring import EnhancedMonitor


class SimulationVisualizer:

    """Real-time and post-simulation visualization system."""


    def __init__(self):

        self.figure, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        plt.ion()
        self.setup_plots()
        self.stages = ['Waiting', 'Preparation', 'Operation', 'Recovery']

    
    def setup_plots(self):
        
        self.axes[0, 0].set_title('Resource Utilization Over Time')
        self.axes[0, 1].set_title('Queue Lengths')
        self.axes[1, 0].set_title('Patient Flow Through System')
        self.axes[1, 1].set_title('Waiting Time Distribution')
        
        for ax in self.axes.flat:
            ax.grid(True)

    
    def update_plots(self, monitor: EnhancedMonitor):

        self._plot_resource_utilization(self.axes[0, 0], monitor.resource_usage)
        self._plot_queue_lengths(self.axes[0, 1], monitor.metrics['queues'], monitor.metrics['queue_timestamps'])
        self._plot_patient_flow(self.axes[1, 0], monitor.patient_records)
        self._plot_waiting_times(self.axes[1, 1], monitor.patient_records)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    
    def _plot_resource_utilization(self, ax, resource_usage):

        ax.clear()

        for resource, usage_data in resource_usage.items():

            if usage_data:

                times, counts = zip(*usage_data)
                ax.plot(times, counts, label=resource)

        ax.legend()
        ax.set_xlabel('Simulation Time')
        ax.set_ylabel('Resources in Use')
        ax.set_title('Resource Utilization Over Time')

    
    def _plot_queue_lengths(self, ax, queue_data, queue_timestamps):

        ax.clear()
        
        timestamps = queue_timestamps['timestamp']
        
        if queue_data:

            for queue_name, lengths in queue_data.items():

                if lengths:

                    ax.plot(timestamps, lengths, label=queue_name)

            ax.legend()

        ax.set_xlabel('Time')
        ax.set_ylabel('Queue Length')
        ax.set_title('Queue Lengths Over Time')
    

    def _plot_patient_flow(self, ax, patient_records):

        ax.clear()
        
        if not patient_records:

            ax.set_title('Patient Flow (No Data)')
            return            
        
        data = []
        
        for record in patient_records:

            times = [
                record['arrival_time'],
                record.get('prep_start', record['arrival_time']),
                record.get('operation_start', record.get('prep_start', record['arrival_time'])),
                record.get('recovery_start', record.get('operation_start', record.get('prep_start', record['arrival_time']))),
                record.get('recovery_end', record.get('recovery_start', record.get('operation_start', record.get('prep_start', record['arrival_time']))))
            ]
            
            for i in range(len(self.stages)):

                if times[i] is not None and times[i+1] is not None:

                    data.append({
                        'Patient': record['id'],
                        'Stage': self.stages[i],
                        'Start': times[i],
                        'End': times[i+1]
                    })
        
        if data:

            df = pd.DataFrame(data)
            
            colors = sns.color_palette('husl', n_colors=len(self.stages))
            
            for i, stage in enumerate(self.stages):

                stage_data = df[df['Stage'] == stage]

                if not stage_data.empty:

                    ax.barh(y=stage_data['Patient'], 
                           left=stage_data['Start'],
                           width=stage_data['End'] - stage_data['Start'],
                           label=stage,
                           color=colors[i],
                           alpha=0.6)
            
            ax.legend()
            ax.set_xlabel('Time')
            ax.set_ylabel('Patient ID')
        
        ax.set_title('Patient Flow Through System')

    
    def _plot_waiting_times(self, ax, patient_records):

        ax.clear()
        
        if not patient_records:

            ax.set_title('Waiting Time Distribution (No Data)')
            return
            
        waiting_times = {

            'Preparation': [],
            'Operation': [],
            'Recovery': []

        }
        
        for record in patient_records:

            waiting_times['Preparation'].append(record.get('prep_wait_time', 0))
            waiting_times['Operation'].append(record.get('operation_wait_time', 0))
            waiting_times['Recovery'].append(record.get('recovery_wait_time', 0))
        
        df = pd.DataFrame(waiting_times)
        sns.boxplot(data=df, ax=ax)
        ax.set_ylabel('Wait Time (minutes)')
        ax.set_title('Wait Time Distribution by Stage')