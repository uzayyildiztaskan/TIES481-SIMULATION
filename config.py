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
    URGENT_PREP_TIME_FACTOR = 0.7  # Urgent patients get faster preparation
    URGENT_OPERATION_TIME_MEAN_FACTOR = 0.8  # Shorter mean operation time
    URGENT_OPERATION_TIME_VAR_FACTOR = 0.4  # Higher variance in operation time
    URGENT_RECOVERY_TIME_FACTOR = 1.2  # Longer recovery time due to urgency
    
    # Visualization parameters
    PLOT_LIVE_UPDATES: bool = False
    UPDATE_INTERVAL: float = 5.0
    
    # Random seed for reproducibility
    RANDOM_SEED: Optional[int] = 42
    IS_PAIRED: bool = False

    def __post_init__(self):
        
        if self.RANDOM_SEED is not None:
            random.seed(self.RANDOM_SEED)