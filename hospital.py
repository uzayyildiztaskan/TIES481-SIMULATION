import random
import simpy
from typing import Optional, Dict, Any, List
import numpy as np

from patient import Patient, PatientStage
from config import SimulationConfig
from monitoring import EnhancedMonitor
from routing import SmartRouter


class Hospital:

    """Simulates a hospital with preparation, operation, and recovery facilities."""
    
    def __init__(self, env: simpy.Environment, config: SimulationConfig,
                 monitor: EnhancedMonitor, router: SmartRouter):
        
        self.env = env
        self.config = config
        self.monitor = monitor
        self.router = router
        
        self.prep_rooms = simpy.Resource(env, capacity=config.NUM_PREP_ROOMS)
        self.operating_rooms = simpy.Resource(env, capacity=config.NUM_OPERATING_ROOMS)
        self.recovery_rooms = simpy.Resource(env, capacity=config.NUM_RECOVERY_ROOMS)
        
        self.router.hospital_state.update({

            'resources': {
                'preparation': self.prep_rooms,
                'operation': self.operating_rooms,
                'recovery': self.recovery_rooms
            }

        })
        
        self.patient_counter = 0
        self.active_patients: List[Patient] = []

    
    def start_patient_arrivals(self):

        self.env.process(self._generate_arrivals())

    
    def _generate_arrivals(self):

        while True:

            interarrival_time = random.expovariate(1.0 / self.config.MEAN_INTERARRIVAL_TIME)
            yield self.env.timeout(interarrival_time)
            
            self.patient_counter += 1
            urgent = random.random() < self.config.URGENT_PATIENT_RATIO
            
            patient = Patient(
                id=self.patient_counter,
                urgent=urgent,
                arrival_time=self.env.now
            )
            
            self.active_patients.append(patient)
            self.env.process(self._process_patient(patient))
            
            self._update_monitor()

    
    def _process_patient(self, patient: Patient):

        with self.prep_rooms.request() as prep_request:
            yield prep_request
            patient.start_preparation(self.env.now)
            
            prep_time = random.expovariate(1.0 / self.config.MEAN_PREP_TIME)
            yield self.env.timeout(prep_time)
            patient.end_preparation(self.env.now)
        
        with self.operating_rooms.request() as op_request:

            yield op_request
            patient.start_operation(self.env.now)
            
            operation_time = random.expovariate(1.0 / self.config.MEAN_OPERATION_TIME)
            yield self.env.timeout(operation_time)
            patient.end_operation(self.env.now)
        
        with self.recovery_rooms.request() as recovery_request:

            yield recovery_request
            patient.start_recovery(self.env.now)
            
            recovery_time = random.expovariate(1.0 / self.config.MEAN_RECOVERY_TIME)
            yield self.env.timeout(recovery_time)
            patient.end_recovery(self.env.now)
        
        self.active_patients.remove(patient)
        self._record_patient_completion(patient)

    
    def _update_monitor(self):

        current_time = self.env.now
        
        self.monitor.record_resource_usage(current_time, 'prep_rooms',
                                         len(self.prep_rooms.users))
        self.monitor.record_resource_usage(current_time, 'operating_rooms',
                                         len(self.operating_rooms.users))
        self.monitor.record_resource_usage(current_time, 'recovery_rooms',
                                         len(self.recovery_rooms.users))
       
        
        self._check_bottlenecks()
        
        self.router.hospital_state['current_time'] = current_time

    
    def _check_bottlenecks(self):

        bottleneck_threshold = 0.8
        
        resources = {
            'preparation': self.prep_rooms,
            'operation': self.operating_rooms,
            'recovery': self.recovery_rooms
        }
        self.monitor.record_metric(self.env.now, 'queue_timestamps', 'timestamp', self.env.now)
        for location, resource in resources.items():

            utilization = len(resource.users) / resource.capacity
            queue_length = len(resource.queue)
            
            self.monitor.record_metric(self.env.now, 'queues', location, queue_length)
            

            if utilization > bottleneck_threshold or queue_length > resource.capacity:

                self.monitor.record_bottleneck(
                    time=self.env.now,
                    location=location,
                    queue_length=queue_length
                )
    

    def _record_patient_completion(self, patient: Patient):

        self.monitor.record_patient(patient.to_dict())
        
        if self.config.DETAILED_MONITORING:
        
            self.monitor.record_metric(
                time=self.env.now,
                category='completion_times',
                metric='total_time',
                value=patient.total_time
            )
            self.monitor.record_metric(
                time=self.env.now,
                category='waiting_times',
                metric='total_wait',
                value=patient.total_wait_time
            )
         