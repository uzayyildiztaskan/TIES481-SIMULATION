from dataclasses import dataclass
from typing import Optional, Dict, Any
import random


@dataclass
class SimulationConfig:

    """Enhanced configuration parameters for the simulation."""

    # Time parameters
    IGNORE_TIME: float = 500.0
    SIMULATION_TIME: float = 1500.0
    MEAN_INTERARRIVAL_TIME: float = 25.0
    MEAN_PREP_TIME: float = 40.0
    MEAN_OPERATION_TIME: float = 20.0
    MEAN_RECOVERY_TIME: float = 40.0
    
    # Resource capacities
    NUM_PREP_ROOMS: int = 3
    NUM_OPERATING_ROOMS: int = 1
    NUM_RECOVERY_ROOMS: int = 3
    
    # Monitoring parameters
    MONITOR_INTERVAL: float = 1.0
    DETAILED_MONITORING: bool = True
    
    # Patient routing parameters
    URGENT_PATIENT_RATIO: float = 0.1
    URGENT_PRIORITY_FACTOR: int = 2
    
    # Visualization parameters
    PLOT_LIVE_UPDATES: bool = True
    UPDATE_INTERVAL: float = 5.0
    
    # Random seed for reproducibility
    RANDOM_SEED: Optional[int] = 42

    def __post_init__(self):
        
        if self.RANDOM_SEED is not None:
            random.seed(self.RANDOM_SEED)