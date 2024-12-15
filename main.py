import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, Any, Optional, List
import simpy
import scipy.stats
import random as rand

from config import SimulationConfig
from hospital import Hospital
from monitoring import EnhancedMonitor
from visualization import SimulationVisualizer
from routing import SmartRouter
from regression_configs import REG_CONFIGURATIONS, REG_CODED_CONFIGURATIONS

import pickle

def blocking_time_calculation(metrics, ignore_time, end_timestamp):

    # Type enforcement, maybe a better way exists, but I do not know (hans)
    timestamps = [float(i) for i in metrics['queue_timestamps']['timestamp']]
    operation_queues = [int(i) for i in metrics['queues']['operation']]
    recovery_queues = [int(i) for i in metrics['queues']['recovery']]
    
    operation_status = list(zip(timestamps, operation_queues, recovery_queues))
    steady_status = [event for event in operation_status if event[0] > ignore_time]
    
    # Figure out what the state at time ignore_time was by checking the 
    # last element to be left out of the steady_status, add that to start of 
    # events list, but with timestamp at exactly ignore_time
    state_at_ignore_time = operation_status[-len(steady_status)-1]
    steady_status = [(ignore_time, state_at_ignore_time[1], state_at_ignore_time[2])] + steady_status
    # Also we might have a case where the system ends at a blocked state, 
    # thus the timer should be left to run until that point. We handle
    # this case by adding a last_state at end_timestamp
    last_state = steady_status[-1]
    if last_state[0] > end_timestamp: raise ValueError
    end_state = (end_timestamp, last_state[1], last_state[2])
    steady_status = steady_status + [end_state]
    timer_total = 0
    
    timer_on = False 
    # Having this True should not affect the calculation, since the first event
    # will be compared against the same timestamp as itself, thus contributing
    # a net zero to timer_total
    
    previous_timestamp = ignore_time
    for event in steady_status:
        current_timestamp = event[0]
        persons_in_operation_queue = event[1]
        persons_in_recovery_queue = event[2]
        if timer_on:
            timer_total += (current_timestamp - previous_timestamp)
        previous_timestamp = current_timestamp
        if persons_in_operation_queue and persons_in_recovery_queue: 
            timer_on = True
        else:
            timer_on = False
    
    return timer_total

def queue_average_for_simulation(queues, timestamps, ignore_time, end_timestamp):

        # Type enforcement, maybe a better way exists, but I do not know (hans)

        timestamps = [float(i) for i in timestamps['timestamp']]
        queues = [int(i) for i in queues]

        queue_status = list(zip(list(timestamps), list(queues)))
        
        steady_status = [event for event in queue_status if event[0] > ignore_time]
        # Figure out what the state at time ignore_time was by checking the 
        # last element to be left out of the steady_status, add that to start of 
        # events list, but with timestamp at exactly ignore_time
        state_at_ignore_time = queue_status[-len(steady_status)-1]
        steady_status = [(ignore_time, state_at_ignore_time[1])] + steady_status
        # Also we might have a case where the system ends at a blocked state, 
        # thus the timer should be left to run until that point. We handle
        # this case by adding a last_state at end_timestamp
        last_state = steady_status[-1]
        if last_state[0] > end_timestamp: raise ValueError
        end_state = (end_timestamp, last_state[1])
        steady_status = steady_status + [end_state]
        
        previous_timestamp = ignore_time 
        queue_mass = 0
        # This in effect integration for a step function
        
        for event in steady_status:
            current_timestamp = event[0]
            queue_mass += event[1] * (current_timestamp - previous_timestamp)
            previous_timestamp = current_timestamp
        
        return queue_mass / (end_timestamp - ignore_time)


class SimulationRunner:
    
    
    def __init__(self, config: SimulationConfig):

        self.config = config
        self.monitor = EnhancedMonitor()
        self.visualizer = SimulationVisualizer() if self.config.PLOT_LIVE_UPDATES else SimulationVisualizer(enable_plotting=False)
        self.results_dir = Path("simulation_results")
        self.results_dir.mkdir(exist_ok=True)
    

    def run(self) -> Dict[str, Any]:
        
        print("Starting hospital simulation...")
        start_time = datetime.now()
        
        env = simpy.Environment()
        
        hospital_state = {
            'current_time': 0,
            'resources': {},
            'avg_service_times': {
                'preparation': self.config.MEAN_PREP_TIME,
                'operation': self.config.MEAN_OPERATION_TIME,
                'recovery': self.config.MEAN_RECOVERY_TIME
            }
        }
        
        router = SmartRouter(hospital_state)
        hospital = Hospital(env, self.config, self.monitor, router)
        
        if self.config.PLOT_LIVE_UPDATES:
            env.process(self._update_visualization(env))
        
        hospital.start_patient_arrivals()
        
        env.run(until=self.config.SIMULATION_TIME)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        results = self._compile_results(execution_time)
        self._save_results(results)
        
        return results
    

    def _update_visualization(self, env):
        
        while True:

            yield env.timeout(self.config.UPDATE_INTERVAL)
            self.visualizer.update_plots(self.monitor)

    
    def _compile_results(self, execution_time: float) -> Dict[str, Any]:
        
        prep_queues = self.monitor.metrics['queues']['preparation']
        summary_stats = self.monitor.get_summary_statistics()
        queue_timestamps = self.monitor.metrics['queue_timestamps']

        results = {
            'simulation_config': self.config.__dict__,
            'execution_time': execution_time,
            'summary_statistics': summary_stats,
            'performance_metrics': {
                'throughput': summary_stats['patient_throughput'],
                'avg_wait_times': summary_stats['waiting_times'],
                'resource_utilization': summary_stats['resource_utilization'],
                'bottlenecks': summary_stats['bottleneck_analysis']
            },
            'timestamp': datetime.now().isoformat(),
            'queue_data': self.monitor.metrics['queues'],
            'queue_timestamps': queue_timestamps,
            'preparation_queue_average_length': queue_average_for_simulation(self.monitor.metrics['queues']['preparation'], self.monitor.metrics['queue_timestamps'], SimulationConfig.IGNORE_TIME, 1500), #SimulationConfig.IGNORE_TIME was 1000?
            'blocking_time': blocking_time_calculation(self.monitor.metrics, SimulationConfig.IGNORE_TIME, SimulationConfig.SIMULATION_TIME)
        }
       
        return results
    
    
    def _save_results(self, results: Dict[str, Any]):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = self.results_dir / f"simulation_results_{timestamp}.json"

        with open(results_file, 'w') as f:

            json.dump(results, f, indent=2)
        
        self._save_visualizations(timestamp)
        
        print(f"Results saved to {results_file}")

    
    def _save_visualizations(self, timestamp: str):

        plots_dir = self.results_dir / f"plots_{timestamp}"
        plots_dir.mkdir(exist_ok=True)
        
        self.visualizer.figure.savefig(plots_dir / "overview.png")
        
        self._generate_additional_plots(plots_dir)

    
    def _generate_additional_plots(self, plots_dir: Path):

        plt.close('all')
        
        plt.figure(figsize=(15, 8))
        try:
            self._plot_patient_timeline()
            plt.savefig(plots_dir / "patient_timeline.png")
        finally:
            plt.close()
        
        """"
        plt.figure(figsize=(10, 6))
        self._plot_resource_heatmap()
        plt.savefig(plots_dir / "resource_heatmap.png")
        """

        plt.figure(figsize=(12, 6))
        try:
            self._plot_wait_distributions()
            plt.savefig(plots_dir / "wait_distributions.png")
        finally:
            plt.close() 

    
    def _plot_patient_timeline(self):
        
        data = pd.DataFrame(self.monitor.patient_records)
        
        plt.title("Patient Flow Timeline")
        sns.scatterplot(data=data, x='arrival_time', y='total_time',
                       hue='urgent', style='current_stage')
        plt.xlabel("Arrival Time")
        plt.ylabel("Total Time in System")
        plt.legend(title="Patient Type")
    

    def _plot_resource_heatmap(self):       
        
        """
        Plot resource utilization heatmap with improved data structure
        """
        resource_data = self.monitor.get_summary_statistics()['resource_utilization']
        
        # Prepare data for heatmap
        resources = list(resource_data.keys())
        metrics = ['mean', 'max', 'min', 'std']
        
        data = np.zeros((len(resources), len(metrics)))
        for i, resource in enumerate(resources):
            for j, metric in enumerate(metrics):
                data[i, j] = resource_data[resource][metric]
        
        plt.figure(figsize=(10, 6))
        plt.title("Resource Utilization Heatmap")
        
        # Create heatmap with annotations
        sns.heatmap(data, annot=True, fmt='.2f',
                    xticklabels=metrics,
                    yticklabels=resources,
                    cmap='YlOrRd')
        
        plt.xlabel("Metric")
        plt.ylabel("Resource")
    

    def _plot_wait_distributions(self):
        
        """
        Plot waiting time distributions with improved data handling
        """
        wait_times = self.monitor.get_summary_statistics()['waiting_times']
        
        # Prepare data for plotting
        data = []
        for stage, stats in wait_times.items():
            data.append({
                'Stage': stage,
                'Wait Time': stats['mean'],
                'Std Dev': stats['std']
            })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        plt.title("Waiting Time Distributions by Stage")
        
        # Create bar plot with error bars
        bars = plt.bar(df['Stage'], df['Wait Time'], yerr=df['Std Dev'],
                    capsize=5, alpha=0.8)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
        
        plt.xlabel("Stage")
        plt.ylabel("Average Wait Time (minutes)")
        plt.grid(True, alpha=0.3)

def run_multiple_configurations(base_config: SimulationConfig, configurations: List[Dict[str, Any]], runs_per_config: int = 20):
    """
    Run simulations across multiple configurations
    
    Args:
        base_config (SimulationConfig): Base configuration to clone and modify
        configurations (List[Dict]): List of configuration modifications
        runs_per_config (int): Number of runs for each configuration
    
    Returns:
        Dict containing results for each configuration
    """
    all_configuration_results = {}
    
    for config_name, config_mods in configurations.items():
        # Create a copy of the base configuration
        current_config = SimulationConfig(**{**base_config.__dict__, **config_mods})
        
        config_results = []
        for i in range(runs_per_config):
            runner = SimulationRunner(current_config)
            results = runner.run()
            config_results.append(results)

            print_results_summary(results)
            
            if base_config.PLOT_LIVE_UPDATES:
                plt.show()
                input("Press Enter to continue...")
            
            # Increment random seed for each run
            if current_config.IS_PAIRED:
                current_config.RANDOM_SEED += 1
            else:
                current_config.RANDOM_SEED = rand.randint(0,100000)
        
        all_configuration_results[config_name] = config_results
    
    return all_configuration_results

def run_single_configuration(base_config: SimulationConfig, runs: int = 20):
    """
    Run simulations for a single configuration with multiple runs
    
    Args:
        base_config (SimulationConfig): Configuration to run
        runs (int): Number of runs for the configuration
    
    Returns:
        List containing results from each run
    """
    config_results = []
    
    # Create a copy of the base configuration to avoid modifying the original
    current_config = SimulationConfig(**base_config.__dict__)

    
    for i in range(runs):
        runner = SimulationRunner(current_config)
        results = runner.run()
        config_results.append(results)
        
        if base_config.PLOT_LIVE_UPDATES:
            plt.show()
            input("Press Enter to continue...")
        
        # Increment random seed for each run
        current_config.RANDOM_SEED += 1
    
    return config_results



def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Hospital Queue Simulation")
    
    parser.add_argument("--simulation-time", type=float, default=1500.0,
                       help="Total simulation time")
    parser.add_argument("--prep-rooms", type=int, default=3,
                       help="Number of preparation rooms")
    parser.add_argument("--operating-rooms", type=int, default=1,
                       help="Number of operating rooms")
    parser.add_argument("--recovery-rooms", type=int, default=3,
                       help="Number of recovery rooms")
    parser.add_argument("--urgent-ratio", type=float, default=0.1,
                       help="Ratio of urgent patients")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no-visualization", action="store_true",
                       help="Disable live visualization")
    parser.add_argument("--detailed-monitoring", action="store_true",
                       help="Enable detailed monitoring")
    parser.add_argument("--runs", type=int, default=20,
                       help="Run multiple simulations")
    parser.add_argument("--pickle", type=str, default="simulation_results",
                       help="Filename for results pickle file")
    parser.add_argument("--preset-config", action="store_true",
                       help="Runs the simulation with the preset configurations set for task 3")
    parser.add_argument("--preset-config-pairwise", action="store_true",
                       help="Runs the simulation with the preset configurations pairwise for task 3")
    parser.add_argument("--preset-config-regression", action="store_true",
                       help="Runs the simulation with the preset configurations and does the regression analysis thing.")
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    
    return SimulationConfig(
        SIMULATION_TIME=args.simulation_time,
        NUM_PREP_ROOMS=args.prep_rooms,
        NUM_OPERATING_ROOMS=args.operating_rooms,
        NUM_RECOVERY_ROOMS=args.recovery_rooms,
        URGENT_PATIENT_RATIO=args.urgent_ratio,
        RANDOM_SEED=args.seed,
        PLOT_LIVE_UPDATES=not args.no_visualization,
        DETAILED_MONITORING=args.detailed_monitoring
    )

def create_preset_config() -> SimulationConfig:

    return SimulationConfig(
        SIMULATION_TIME=1500,
        NUM_OPERATING_ROOMS=1,
        RANDOM_SEED=42,
        PLOT_LIVE_UPDATES=False,
        DETAILED_MONITORING=False,
        IS_PAIRED=False
    )

def create_preset_config_pairwise() -> SimulationConfig:

    return SimulationConfig(
        SIMULATION_TIME=1500,
        NUM_OPERATING_ROOMS=1,
        RANDOM_SEED=42,
        PLOT_LIVE_UPDATES=False,
        DETAILED_MONITORING=False,
        IS_PAIRED=True
    )


def print_results_summary(results: Dict[str, Any]):


    
    print("\n=== Simulation Results Summary ===")
    print(f"\nExecution Time: {results['execution_time']:.2f} seconds")
    
    print("\nThroughput:")
    print(f"  Total Patients: {results['performance_metrics']['throughput']['total']}")
    print(f"  Patients per Hour: {results['performance_metrics']['throughput']['per_hour']:.2f}")
    
    print("\nAverage Wait Times:")
    for stage, stats in results['performance_metrics']['avg_wait_times'].items():

        print(f"  {stage}:")
        print(f"    Mean: {stats['mean']:.2f} minutes")
        print(f"    Median: {stats['median']:.2f} minutes")
        print(f"    Sum: {stats['sum']:.2f} minutes")
    
    print("\nResource Utilization:")
    for resource, stats in results['performance_metrics']['resource_utilization'].items():
        if (resource == "prep_rooms"):
            print(f"  {resource}:")
            print(f"    Mean: {stats['mean']/SimulationConfig.NUM_PREP_ROOMS:.2%}")
            print(f"    Max: {stats['max']/SimulationConfig.NUM_PREP_ROOMS:.2%}")
        elif (resource == "recovery_rooms"):
            print(f"  {resource}:")
            print(f"    Mean: {stats['mean']/SimulationConfig.NUM_RECOVERY_ROOMS:.2%}")
            print(f"    Max: {stats['max']/SimulationConfig.NUM_RECOVERY_ROOMS:.2%}")
        else:
            print(f"  {resource}:")
            print(f"    Mean: {stats['mean']:.2%}")
            print(f"    Max: {stats['max']:.2%}")
            
    print("\nBottleneck Analysis:")
    for location, stats in results['performance_metrics']['bottlenecks'].items():

        print(f"  {location}:")
        print(f"    Occurrences: {stats['count']}")
        print(f"    Max Queue: {stats['max_queue']}")



def main():

    args = parse_arguments()

    save_path = args.pickle

    if args.preset_config:

        base_config = create_preset_config()
    
        configurations = {
            "3p3r": {
                "NUM_PREP_ROOMS": 3,
                "NUM_RECOVERY_ROOMS": 3
            },
            "3p4r": {
                "NUM_PREP_ROOMS": 3,
                "NUM_RECOVERY_ROOMS": 4
            },
            "4p5r": {
                "NUM_PREP_ROOMS": 4,
                "NUM_RECOVERY_ROOMS": 5
            }
        }

        results = run_multiple_configurations(base_config, configurations)

    elif args.preset_config_pairwise:

        base_config = create_preset_config_pairwise()
    
        configurations = {
            "base": {
                "NUM_PREP_ROOMS": 3,
                "NUM_RECOVERY_ROOMS": 3,
                "INTERARRIVAL_TIME": "exp(25)",
                "PREPARATION_TIME": "exp(40)",
                "RECOVERY_TIME": "exp(40)"
            },
            "extra prep": {
                "NUM_PREP_ROOMS": 4,
                "NUM_RECOVERY_ROOMS": 3,
                "INTERARRIVAL_TIME": "exp(25)",
                "PREPARATION_TIME": "exp(40)",
                "RECOVERY_TIME": "exp(40)"
            }
        }
        
        results = run_multiple_configurations(base_config, configurations)

    elif args.preset_config_pairwise:

        base_config = create_preset_config_pairwise()
    
        configurations = {
            "base": {
                "NUM_PREP_ROOMS": 3,
                "NUM_RECOVERY_ROOMS": 3,
                "INTERARRIVAL_TIME": "exp(25)",
                "PREPARATION_TIME": "exp(40)",
                "RECOVERY_TIME": "exp(40)"
            },
            "extra prep": {
                "NUM_PREP_ROOMS": 4,
                "NUM_RECOVERY_ROOMS": 3,
                "INTERARRIVAL_TIME": "exp(25)",
                "PREPARATION_TIME": "exp(40)",
                "RECOVERY_TIME": "exp(40)"
            }
        }
        
        results = run_multiple_configurations(base_config, configurations)
    elif args.preset_config_regression:
        base_config = create_preset_config()

        configurations = REG_CONFIGURATIONS

        results = run_multiple_configurations(base_config, configurations)
    else:

        base_config = create_config_from_args(args)

        results = run_single_configuration(base_config, runs=args.runs)
    
    
    # Save results to pickle
    with open(f'{save_path}/multi_config_simulation.pkl', 'wb') as f:
        pickle.dump(results, f)

    
    
    
    
    # Optional: Comparative Statistical Test
    if args.preset_config:
        comparative_statistical_test(results)
    # Pairwise test (Note: doing this at the same time as the above produces nonsense
    # as it needs a random seed and this needs a fixed seed)
    if args.preset_config_pairwise:
        pairwise_statistical_test(results)
        # Optional: Generate comparative report
        generate_comparative_report(results)

    if args.preset_config_regression:
        regression_analysis(results)

def generate_comparative_report(results: Dict[str, List[Dict]]):
    """
    Generate a comprehensive report comparing simulation configurations
    
    Args:
        results (Dict): Results from multiple configuration runs
    """
    print("\n=== CONFIGURATION COMPARATIVE ANALYSIS ===")
    
    for config_name, config_runs in results.items():
        print(f"\nConfiguration: {config_name}")
        
        # Aggregate metrics across runs
        throughputs = [run['performance_metrics']['throughput']['per_hour'] for run in config_runs]
        wait_times = [np.mean([stage['mean'] for stage in run['performance_metrics']['avg_wait_times'].values()]) 
                      for run in config_runs]
        preparation_utils = [float(run['performance_metrics']['resource_utilization']['prep_rooms']['mean'])
                          for run in config_runs]
        
        print(f"Throughput (patients/hour): {np.mean(throughputs):.2f} ± {np.std(throughputs):.2f}")
        print(f"Average Wait Time: {np.mean(wait_times):.2f} ± {np.std(wait_times):.2f} minutes")
        print(f"Preparation Room Utilization: {np.mean(preparation_utils):.2%} ± {np.std(preparation_utils):.2%}")
        
        
        # Bottleneck analysis
        bottlenecks = [run['performance_metrics']['bottlenecks'] for run in config_runs]
        print("Bottleneck Summary:")
        for location in set().union(*bottlenecks):
            location_bottlenecks = [b.get(location, {'count': 0, 'max_queue': 0}) for b in bottlenecks]
            print(f"  {location}:")
            print(f"    Avg Bottleneck Count: {np.mean([b['count'] for b in location_bottlenecks]):.2f}")
            print(f"    Max Queue Length: {np.mean([b['max_queue'] for b in location_bottlenecks]):.2f}")

# Helper function: Compute confidence intervals
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    lower, upper = scipy.stats.t.interval(
        confidence, len(data) - 1, loc=mean, scale=scipy.stats.sem(data)
    )
    return mean, lower, upper

def extract_avg_queue_lengths(results):
    """
    Extracts the overall average preparation queue length for each configuration.
    """
    avg_queue_lengths = {}
    for config_name, runs in results.items():
        # Extract the list of queue lengths for all runs
        queue_lengths = average_queue_lengths_from_data(
            runs, 'preparation', SimulationConfig.IGNORE_TIME, SimulationConfig.SIMULATION_TIME
        )
        # Calculate the mean of the queue lengths
        avg_queue_lengths[config_name] = np.mean(queue_lengths)
    return avg_queue_lengths

def regression_analysis(results: Dict[str, List[Dict]]):
    # Simple Linear Regression using coded values
    import statsmodels.api as sm
    
    avg_queue_lengths = extract_avg_queue_lengths(results)
    # Constructing a dataframe for regression
    data = []
    for config_name, avg_length in avg_queue_lengths.items():
        row = list(REG_CODED_CONFIGURATIONS[config_name].values()) + [avg_length]
        data.append(row)

    
    df = pd.DataFrame(data, columns=["A_Interarrival_Distribution", "B_Urgent_Ratio", "C_Preparation_Time", 
                                    "D_Recovery_Time", "E_Preparation_Units", "F_Recovery_Units", "Queue_Length"])
    
    # Set up the model
    X = df.drop(columns=["Queue_Length"])
    X = sm.add_constant(X)  # Adds the constant term for the intercept
    y = df["Queue_Length"]
    
    model = sm.OLS(y, X).fit()  # Ordinary Least Squares regression
    
    # Print regression results
    print(model.summary())

# Example of how to pull results and run regression
# Let's assume `results` is a dictionary where each key is a configuration name, and the value is the simulation result
results = {
    "base": [{"performance_metrics": {"queue_length": 5}}],  # Replace with actual simulation data
    "extra prep": [{"performance_metrics": {"queue_length": 7}}],  # Replace with actual simulation data
    # Add more configurations here
}



def comparative_statistical_test(results: Dict[str, List[Dict]]):
    """
    Perform statistical tests to compare configurations
    """
    from scipy import stats
    import numpy as np
    
    configurations = list(results.keys())
    print("\n=== Comparative Statistical Analysis ===")
    
    # Metrics extraction and comparison
    metrics = {
        "Throughput (patients/hour)": lambda runs: [run['performance_metrics']['throughput']['per_hour'] for run in runs],
        "Average Wait Time (minutes)": lambda runs: [np.mean([stage['mean'] for stage in run['performance_metrics']['avg_wait_times'].values()]) for run in runs],
        "Preparation Room Utilization": lambda runs: [float(run['performance_metrics']['resource_utilization']['prep_rooms']['mean']) for run in runs],
        "Blocking Probability": lambda runs: [bt / (SimulationConfig.SIMULATION_TIME) for bt in operation_blocking_time_from_data(runs, SimulationConfig.IGNORE_TIME, SimulationConfig.SIMULATION_TIME)],
        "Average Preparation Queue Length": lambda runs: [pq for pq in average_queue_lengths_from_data(runs, 'preparation', SimulationConfig.IGNORE_TIME, SimulationConfig.SIMULATION_TIME)]
    }
    
    for metric_name, metric_func in metrics.items():
        print(f"\nComparative Analysis for {metric_name}:")
        for i in range(len(configurations)):
            for j in range(i+1, len(configurations)):
                config1, config2 = configurations[i], configurations[j]
                values1 = metric_func(results[config1])
                values2 = metric_func(results[config2])
                
                t_statistic, p_value = stats.ttest_ind(values1, values2)
                mean1, lower1, upper1 = confidence_interval(values1)
                mean2, lower2, upper2 = confidence_interval(values2)
                
                print(f"  {config1} vs {config2}:")
                print(f"    {config1} Mean: {mean1:.2f}, 95% CI: [{lower1:.2f}, {upper1:.2f}]")
                print(f"    {config2} Mean: {mean2:.2f}, 95% CI: [{lower2:.2f}, {upper2:.2f}]")
                print(f"    t-statistic: {t_statistic:.4f}, p-value: {p_value:.4f}")
                print(f"    Significant Difference: {'Yes' if p_value < 0.05 else 'No'}")

def pairwise_statistical_test(results: Dict[str, List[Dict]]):
    """
    Perform paired statistical tests to compare configurations
    """
    configurations = list(results.keys())
    metrics = {
        "Blocking Probability": lambda run: operation_blocking_time_from_data(run, SimulationConfig.IGNORE_TIME, SimulationConfig.SIMULATION_TIME),
        "Preparation Queue Length": lambda run: average_queue_lengths_from_data(run, "preparation", SimulationConfig.IGNORE_TIME, SimulationConfig.SIMULATION_TIME),
        "Preparation Room Utilization": lambda run: [sim["performance_metrics"]["resource_utilization"]["prep_rooms"]["mean"] for sim in run]
    }

    for metric_name, metric_func in metrics.items():
        print(f"\nComparative Analysis for {metric_name} (Paired):")
        for i in range(len(configurations)):
            for j in range(i+1, len(configurations)):
                config1, config2 = configurations[i], configurations[j]
                values1 = metric_func(results[config1])
                values2 = metric_func(results[config2])
                
                # Compute differences
                differences = np.array(values1) - np.array(values2)
                t_statistic, p_value = scipy.stats.ttest_rel(values1, values2)
                mean_diff = np.mean(differences)
                lower, upper = scipy.stats.t.interval(0.95, len(differences)-1, loc=mean_diff, scale=scipy.stats.sem(differences))
                
                print(f"  {config1} vs {config2}:")
                print(f"    Mean Difference: {mean_diff:.4f}")
                print(f"    95% CI for Difference: [{lower:.4f}, {upper:.4f}]")
                print(f"    t-statistic: {t_statistic:.4f}, p-value: {p_value:.4f}")
                print(f"    Significant Difference: {'Yes' if p_value < 0.05 else 'No'}")

def operation_blocking_time_from_data(data, ignore_time, end_timestamp):
    # In order to estimate from a steady state, use ignore_time
    # to ignore events before timestamp = ignore_time
    if end_timestamp < ignore_time: raise ValueError
 
    timer_totals = []
    for simulation in data:
        operation_status = list(zip(simulation['queue_timestamps']['timestamp'], 
                               simulation['queue_data']['operation'],
                               simulation['queue_data']['recovery']))
        steady_status = [event for event in operation_status if event[0] > ignore_time]
        # Figure out what the state at time ignore_time was by checking the 
        # last element to be left out of the steady_status, add that to start of 
        # events list, but with timestamp at exactly ignore_time
        state_at_ignore_time = operation_status[-len(steady_status)-1]
        steady_status = [(ignore_time, state_at_ignore_time[1], state_at_ignore_time[2])] + steady_status
        # Also we might have a case where the system ends at a blocked state, 
        # thus the timer should be left to run until that point. We handle
        # this case by adding a last_state at end_timestamp
        last_state = steady_status[-1]
        if last_state[0] > end_timestamp: raise ValueError
        end_state = (end_timestamp, last_state[1], last_state[2])
        steady_status = steady_status + [end_state]
        timer_total = 0
        
        timer_on = False 
        # Having this True should not affect the calculation, since the first event
        # will be compared against the same timestamp as itself, thus contributing
        # a net zero to timer_total
        
        previous_timestamp = ignore_time
        for event in steady_status:
            current_timestamp = event[0]
            persons_in_operation_queue = event[1]
            persons_in_recovery_queue = event[2]
            if timer_on:
                timer_total += (current_timestamp - previous_timestamp)
            previous_timestamp = current_timestamp
            if persons_in_operation_queue and persons_in_recovery_queue: 
                timer_on = True
            else:
                timer_on = False
                
        timer_totals.append(timer_total)
        
    return timer_totals

def preparation_capacities(data):
    return [simulation['summary_statistics']['resource_utilization']['prep_rooms']['mean'] for simulation in data]
    
def average_queue_lengths_from_data(data, location, ignore_time, end_timestamp):
    # In order to estimate from a steady state, use ignore_time
    # to ignore events before timestamp = ignore_time
    if end_timestamp < ignore_time: raise ValueError
    queue_averages = []
    for simulation in data:
        queue_status = list(zip(simulation['queue_timestamps']['timestamp'], simulation['queue_data'][location]))
        steady_status = [event for event in queue_status if event[0] > ignore_time]
        # Figure out what the state at time ignore_time was by checking the 
        # last element to be left out of the steady_status, add that to start of 
        # events list, but with timestamp at exactly ignore_time
        state_at_ignore_time = queue_status[-len(steady_status)-1]
        steady_status = [(ignore_time, state_at_ignore_time[1])] + steady_status
        # Also we might have a case where the system ends at a blocked state, 
        # thus the timer should be left to run until that point. We handle
        # this case by adding a last_state at end_timestamp
        last_state = steady_status[-1]
        if last_state[0] > end_timestamp: raise ValueError
        end_state = (end_timestamp, last_state[1])
        steady_status = steady_status + [end_state]
        
        previous_timestamp = ignore_time 
        queue_mass = 0
        # This in effect integration for a step function
        
        for event in steady_status:
            current_timestamp = event[0]
            queue_mass += event[1] * (current_timestamp - previous_timestamp)
            previous_timestamp = current_timestamp
        
        queue_averages.append(queue_mass / (end_timestamp - ignore_time))
   
    return queue_averages
    


if __name__ == "__main__":
    main()