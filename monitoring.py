from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime


class EnhancedMonitor:

    def __init__(self):

        self.timestamps: List[float] = []
        self.metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.patient_records: List[Dict[str, Any]] = []
        self.resource_usage: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
        self.bottleneck_events: List[Dict[str, Any]] = []

        
    def record_metric(self, time: float, category: str, metric: str, value: float):

        self.timestamps.append(time)
        self.metrics[category][metric].append(value)

    
    def record_patient(self, patient_data: Dict[str, Any]):
        
        self.patient_records.append(patient_data)

    
    def record_resource_usage(self, time: float, resource: str, in_use: int):
        
        self.resource_usage[resource].append((time, in_use))

    
    def record_bottleneck(self, time: float, location: str, queue_length: int):
        
        self.bottleneck_events.append({
            'time': time,
            'location': location,
            'queue_length': queue_length
        })

    
    def get_summary_statistics(self) -> Dict[str, Any]:
        
        stats = {
            'patient_throughput': {
                'total': len(self.patient_records),
                'per_hour': len(self.patient_records) / (max(self.timestamps) / 60) if self.timestamps else 0
            },
            'waiting_times': self._calculate_waiting_times(),
            'resource_utilization': self._calculate_resource_utilization(),
            'bottleneck_analysis': self._analyze_bottlenecks()
        }

        return stats
    
    
    def _calculate_waiting_times(self) -> Dict[str, Dict[str, float]]:
        
        waiting_times = defaultdict(list)

        for record in self.patient_records:

            for phase in ['prep', 'operation', 'recovery']:

                wait_time = record.get(f'{phase}_wait_time', 0)
                waiting_times[phase].append(wait_time)
        
        return {
            phase: {
                'mean': np.mean(times),
                'median': np.median(times),
                'std': np.std(times),
                'max': max(times),
                'sum': sum(times)
            }
            for phase, times in waiting_times.items()
            if times
        }
    
    
    def _calculate_resource_utilization(self) -> Dict[str, Dict[str, float]]:
        
        utilization_stats = {}

        for resource, usage_data in self.resource_usage.items():

            times, counts = zip(*usage_data)
            utilization_stats[resource] = {
                'mean': np.mean(counts),
                'max': max(counts),
                'min': min(counts),
                'std': np.std(counts)
            }

        return utilization_stats
    

    def _analyze_bottlenecks(self) -> Dict[str, Any]:
        
        if not self.bottleneck_events:

            return {}
        
        bottleneck_analysis = defaultdict(lambda: {'count': 0, 'max_queue': 0})

        for event in self.bottleneck_events:
            
            location = event['location']
            bottleneck_analysis[location]['count'] += 1
            bottleneck_analysis[location]['max_queue'] = max(
                bottleneck_analysis[location]['max_queue'],
                event['queue_length']
            )
        
        return dict(bottleneck_analysis)