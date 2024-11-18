from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum, auto


class PatientStage(Enum):

    ARRIVAL = auto()
    PREPARATION = auto()
    OPERATION = auto()
    RECOVERY = auto()
    DISCHARGED = auto()


@dataclass
class Patient:
    
    id: int
    urgent: bool
    arrival_time: float
    
    # Current state
    current_stage: PatientStage = PatientStage.ARRIVAL
    
    # Timing records
    prep_start: Optional[float] = None
    prep_end: Optional[float] = None
    operation_start: Optional[float] = None
    operation_end: Optional[float] = None
    recovery_start: Optional[float] = None
    recovery_end: Optional[float] = None
    
    # Wait time tracking
    prep_wait_time: float = 0.0
    operation_wait_time: float = 0.0
    recovery_wait_time: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    

    @property
    def total_wait_time(self) -> float:

        return self.prep_wait_time + self.operation_wait_time + self.recovery_wait_time
    
    
    @property
    def total_time(self) -> float:

        if self.recovery_end:

            return self.recovery_end - self.arrival_time
        
        return 0.0
    

    def start_preparation(self, time: float):

        self.prep_wait_time = time - (self.arrival_time if not self.prep_end else self.prep_end)
        self.prep_start = time
        self.current_stage = PatientStage.PREPARATION

    
    def end_preparation(self, time: float):

        self.prep_end = time
    

    def start_operation(self, time: float):

        self.operation_wait_time = time - self.prep_end
        self.operation_start = time
        self.current_stage = PatientStage.OPERATION
    

    def end_operation(self, time: float):

        self.operation_end = time
    
    
    def start_recovery(self, time: float):

        self.recovery_wait_time = time - self.operation_end
        self.recovery_start = time
        self.current_stage = PatientStage.RECOVERY

    
    def end_recovery(self, time: float):
        
        self.recovery_end = time
        self.current_stage = PatientStage.DISCHARGED

    
    def to_dict(self) -> Dict[str, Any]:
        
        return {
            'id': self.id,
            'urgent': self.urgent,
            'arrival_time': self.arrival_time,
            'current_stage': self.current_stage.name,
            'prep_start': self.prep_start,
            'prep_end': self.prep_end,
            'operation_start': self.operation_start,
            'operation_end': self.operation_end,
            'recovery_start': self.recovery_start,
            'recovery_end': self.recovery_end,
            'prep_wait_time': self.prep_wait_time,
            'operation_wait_time': self.operation_wait_time,
            'recovery_wait_time': self.recovery_wait_time,
            'total_wait_time': self.total_wait_time,
            'total_time': self.total_time
        }