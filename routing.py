from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RoutingDecision:

    next_stage: str
    priority: int
    estimated_wait: float


class SmartRouter:

    def __init__(self, hospital_state: Dict[str, Any]):

        self.hospital_state = hospital_state
        self.routing_history: List[Dict[str, Any]] = []
    

    def decide_next_route(self, patient: 'Patient') -> RoutingDecision:

        current_stage = patient.current_stage
        
        stage_loads = self._calculate_stage_loads()
        
        priority = self._calculate_priority(patient)
        
        wait_times = self._estimate_wait_times(stage_loads)
        
        next_stage = self._select_optimal_route(
            current_stage, stage_loads, wait_times, patient.urgent
        )
        
        decision = RoutingDecision(
            next_stage=next_stage,
            priority=priority,
            estimated_wait=wait_times[next_stage]
        )
        
        self._record_decision(patient, decision)
        
        return decision
    
    
    def _calculate_stage_loads(self) -> Dict[str, float]:

        return {
            stage: len(resource.queue) / resource.capacity
            for stage, resource in self.hospital_state['resources'].items()
        }
    
    
    def _calculate_priority(self, patient: 'Patient') -> int:

        base_priority = 1
        if patient.urgent:
            base_priority *= 2
        
        wait_time = patient.total_wait_time

        if wait_time > 60:

            base_priority += 1
        
        return base_priority
    
    
    def _estimate_wait_times(self, loads: Dict[str, float]) -> Dict[str, float]:

        wait_times = {}

        for stage, load in loads.items():

            historical_service_time = self.hospital_state['avg_service_times'][stage]
            wait_times[stage] = load * historical_service_time

        return wait_times
    
    
    def _select_optimal_route(
        self,
        current_stage: str,
        loads: Dict[str, float],
        wait_times: Dict[str, float],
        is_urgent: bool
    ) -> str:
        
        possible_routes = self._get_possible_routes(current_stage)
        
        if is_urgent:

            return min(possible_routes, key=lambda x: wait_times[x])
        
        else:

            return min(
                possible_routes,
                key=lambda x: loads[x] * wait_times[x]
            )
    

    def _get_possible_routes(self, current_stage: str) -> List[str]:

        route_map = {
            'arrival': ['preparation'],
            'preparation': ['operation'],
            'operation': ['recovery'],
            'recovery': ['discharge']
        }

        return route_map.get(current_stage, [])
    
    
    def _record_decision(self, patient: 'Patient', decision: RoutingDecision):
        
        self.routing_history.append({
            'time': self.hospital_state['current_time'],
            'patient_id': patient.id,
            'patient_type': 'urgent' if patient.urgent else 'regular',
            'current_stage': patient.current_stage,
            'next_stage': decision.next_stage,
            'priority': decision.priority,
            'estimated_wait': decision.estimated_wait
        })