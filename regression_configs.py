REG_CONFIGURATIONS = {
    "1": {
        "INTERARRIVAL_TIME": "exp(25)",
        "PREPARATION_TIME": "exp(40)",
        "RECOVERY_TIME": "exp(40)",
        "NUM_PREP_ROOMS": 4,
        "NUM_RECOVERY_ROOMS": 4
    },
    "2": {
        "INTERARRIVAL_TIME": "exp(25)",
        "PREPARATION_TIME": "unif(30,50)",
        "RECOVERY_TIME": "unif(30,50)",
        "NUM_PREP_ROOMS": 5,
        "NUM_RECOVERY_ROOMS": 4
    },
    "3": {
        "INTERARRIVAL_TIME": "exp(22.5)",
        "PREPARATION_TIME": "exp(40)",
        "RECOVERY_TIME": "unif(30,50)",
        "NUM_PREP_ROOMS": 5,
        "NUM_RECOVERY_ROOMS": 4
    },
    "4": {
        "INTERARRIVAL_TIME": "exp(22.5)",
        "PREPARATION_TIME": "unif(30,50)",
        "RECOVERY_TIME": "exp(40)",
        "NUM_PREP_ROOMS": 4,
        "NUM_RECOVERY_ROOMS": 4
    },
    "5": {
        "INTERARRIVAL_TIME": "unif(20,30)",
        "PREPARATION_TIME": "exp(40)",
        "RECOVERY_TIME": "unif(30,50)",
        "NUM_PREP_ROOMS": 4,
        "NUM_RECOVERY_ROOMS": 5
    },
    "6": {
        "INTERARRIVAL_TIME": "unif(20,30)",
        "PREPARATION_TIME": "unif(30,50)",
        "RECOVERY_TIME": "exp(40)",
        "NUM_PREP_ROOMS": 5,
        "NUM_RECOVERY_ROOMS": 5
    },
    "7": {
        "INTERARRIVAL_TIME": "unif(20,25)",
        "PREPARATION_TIME": "exp(40)",
        "RECOVERY_TIME": "exp(40)",
        "NUM_PREP_ROOMS": 5,
        "NUM_RECOVERY_ROOMS": 5
    },
    "8": {
        "INTERARRIVAL_TIME": "unif(20,25)",
        "PREPARATION_TIME": "unif(30,50)",
        "RECOVERY_TIME": "unif(30,50)",
        "NUM_PREP_ROOMS": 4,
        "NUM_RECOVERY_ROOMS": 5
    }
}

def map_to_coded_values(configurations):
    """
    Map configuration parameters to coded values (-1, +1)
    """
    coded_configurations = {}

    for name, config in configurations.items():
        coded_configurations[name] = {
            "A": -1 if "exp" in config["INTERARRIVAL_TIME"] else 1,
            "B": -1 if config["INTERARRIVAL_TIME"] == "exp(25)" else 1,
            "C": -1 if config["PREPARATION_TIME"] == "exp(40)" else 1,
            "D": -1 if config["RECOVERY_TIME"] == "exp(40)" else 1,
            "E": -1 if config["NUM_PREP_ROOMS"] == 4 else 1,
            "F": -1 if config["NUM_RECOVERY_ROOMS"] == 4 else 1,
        }
    
    return coded_configurations

# Automatically generate coded values
REG_CODED_CONFIGURATIONS = map_to_coded_values(REG_CONFIGURATIONS)