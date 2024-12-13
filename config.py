from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import random


@dataclass
class SimulationConfig:

    """Enhanced configuration parameters for the simulation."""

    # Time parameters
    IGNORE_TIME: float = 500.0
    SIMULATION_TIME: float = 1500.0
    MEAN_INTERARRIVAL_TIME: float = 25.0
    INTERARRIVAL_TIME: str = "exp(25)"
    PREPARATION_TIME: str = "exp(40)"
    RECOVERY_TIME: str = "exp(40)"
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

    def get_time_function(self, time_type: str):
        predefined_distributions = {
            "interarrival": {
                "exp(25)": lambda: random.expovariate(1.0 / 25.0),
                "exp(22.5)": lambda: random.expovariate(1.0 / 22.5),
                "unif(20,30)": lambda: random.uniform(20.0, 30.0),
                "unif(20,25)": lambda: random.uniform(20.0, 25.0)
            },
            "preparation": {
                "exp(40)": lambda: random.expovariate(1.0 / 40.0),
                "unif(30,50)": lambda: random.uniform(30.0, 50.0),
            },
            "recovery": {
                "exp(40)": lambda: random.expovariate(1.0 / 40.0),
                "unif(30,50)": lambda: random.uniform(30.0, 50.0),
            }
        }   
    
        # Get the selected distribution based on time_type and config value
        selected_value = getattr(self, f"{time_type.upper()}_TIME", None)
        if selected_value not in predefined_distributions[time_type]:
            raise ValueError(f"Unknown {time_type} time distribution: {selected_value}")
    
        return predefined_distributions[time_type][selected_value]

    def serialize(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Reconstruct a SimulationConfig instance from a dictionary."""
        return cls(**data)