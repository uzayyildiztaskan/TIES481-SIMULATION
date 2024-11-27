import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, Any, Optional
import simpy

from config import SimulationConfig
from hospital import Hospital
from monitoring import EnhancedMonitor
from visualization import SimulationVisualizer
from routing import SmartRouter

import pickle

class SimulationRunner:
    
    
    def __init__(self, config: SimulationConfig):

        self.config = config
        self.monitor = EnhancedMonitor()
        self.visualizer = SimulationVisualizer()
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
        
        summary_stats = self.monitor.get_summary_statistics()
        
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
            'timestamp': datetime.now().isoformat()
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
        
        plt.figure(figsize=(15, 8))
        self._plot_patient_timeline()
        plt.savefig(plots_dir / "patient_timeline.png")
        
        """"
        plt.figure(figsize=(10, 6))
        self._plot_resource_heatmap()
        plt.savefig(plots_dir / "resource_heatmap.png")
        """

        plt.figure(figsize=(12, 6))
        self._plot_wait_distributions()
        plt.savefig(plots_dir / "wait_distributions.png")

    
    def _plot_patient_timeline(self):
        
        data = pd.DataFrame(self.monitor.patient_records)
        
        plt.title("Patient Flow Timeline")
        sns.scatterplot(data=data, x='arrival_time', y='total_time',
                       hue='urgent', style='current_stage')
        plt.xlabel("Arrival Time")
        plt.ylabel("Total Time in System")
        plt.legend(title="Patient Type")
    

    def _plot_resource_heatmap(self):
        
        
        utilization_data = pd.DataFrame(self.monitor.resource_usage)
    
        plt.title("Resource Utilization Heatmap")
        sns.heatmap(utilization_data, cmap='YlOrRd',
                   xticklabels=True, yticklabels=True)
        plt.xlabel("Time Period")
        plt.ylabel("Resource")
    

    def _plot_wait_distributions(self):
        
        wait_times = pd.DataFrame(self.monitor.get_summary_statistics()['waiting_times'])
        
        plt.title("Waiting Time Distributions by Stage")
        sns.boxplot(data=wait_times)
        plt.xlabel("Stage")
        plt.ylabel("Wait Time (minutes)")


def parse_arguments() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Hospital Queue Simulation")
    
    parser.add_argument("--simulation-time", type=float, default=1000.0,
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
    parser.add_argument("--pickle", type=str, default="simulation.results",
                       help="Filename for results pickle file")
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    
    return SimulationConfig(
        SIMULATION_TIME=args.simulation_time,
        NUM_PREP_ROOMS=args.prep_rooms,
        NUM_OPERATING_ROOMS=args.operating_rooms,
        NUM_RECOVERY_ROOMS=args.recovery_rooms,
        URGENT_PATIENT_RATIO=args.urgent_ratio,
        RANDOM_SEED=args.seed,
        PLOT_LIVE_UPDATES=args.no_visualization,
        DETAILED_MONITORING=args.detailed_monitoring
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
    
    try:

        args = parse_arguments()
        
        config = create_config_from_args(args)
        
        all_run_results = []
        for i in range(args.runs):
            runner = SimulationRunner(config)
            results = runner.run()
            
            print_results_summary(results)
            
            if config.PLOT_LIVE_UPDATES:
                plt.show()
                input("Press Enter to continue...")
    
            all_run_results.append(results)
            config.RANDOM_SEED += 1 # Run next simulation with different seed, so we don't run the same simulation over and over again.
        
        file = open(args.pickle, 'wb')

        pickle.dump(all_run_results, file)     
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during simulation: {str(e)}")
        raise


if __name__ == "__main__":
    main()