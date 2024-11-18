# Run with default settings
python -m simulation.main

# Run with custom settings
python -m simulation.main --simulation-time 2000 --prep-rooms 4 --urgent-ratio 0.15 --detailed-monitoring

# Run without visualization
python -m simulation.main --no-visualization
