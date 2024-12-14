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
            'bottleneck_analysis': self._analyze_bottlenecks(),
            'full_patient_records': self.patient_records
        }

        return stats
    
    
    def _calculate_waiting_times(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate waiting time statistics with proper error handling and data validation
        """
        waiting_times = defaultdict(list)
        stages = ['prep', 'operation', 'recovery']
        
        for record in self.patient_records:
            for stage in stages:
                wait_time = record.get(f'{stage}_wait_time', 0)
                if wait_time is not None and wait_time >= 0:  # Data validation
                    waiting_times[stage].append(wait_time)
        
        stats = {}
        for stage, times in waiting_times.items():
            if times:  # Only calculate stats if we have data
                stats[stage] = {
                    'mean': float(np.mean(times)),
                    'median': float(np.median(times)),
                    'std': float(np.std(times)) if len(times) > 1 else 0,
                    'max': float(max(times)),
                    'sum': float(sum(times))
                }
            else:
                stats[stage] = {
                    'mean': 0, 'median': 0, 'std': 0, 'max': 0, 'sum': 0
                }
        
        return stats
    
    
    def _calculate_resource_utilization(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate resource utilization with improved error handling
        """
        utilization_stats = {}
        
        for resource, usage_data in self.resource_usage.items():
            if usage_data:  # Check if we have data
                times, counts = zip(*usage_data)
                counts = [float(c) for c in counts]  # Ensure float type
                
                utilization_stats[resource] = {
                    'mean': float(np.mean(counts)),
                    'max': float(max(counts)),
                    'min': float(min(counts)),
                    'std': float(np.std(counts)) if len(counts) > 1 else 0
                }
            else:
                utilization_stats[resource] = {
                    'mean': 0, 'max': 0, 'min': 0, 'std': 0
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