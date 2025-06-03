# Lane Detection Project

This project implements lane detection using neural networks and computer vision techniques. It includes various applications for training, testing, and real-time video processing.

## Prerequisites

- Python 3.12 or higher
- CUDA-compatible GPU (recommended for training)
- pip (Python package manager)

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd LaneDetectionV2
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `app/`: Contains the main application scripts
- `lane_detector/`: Core lane detection implementation
- `neural_networks/`: Neural network models and training code
- `utils/`: Utility functions and helper modules
- `data/`: Dataset storage
- `outputs/`: Output files and results
- `logs/`: Training and application logs
- `camera/`: Camera interface and related code
- `assets/`: Static assets and resources

## Usage

The project provides several commands through the Makefile:

### Training

```bash
make train
```

Trains the ENet model for lane detection.

### Testing

```bash
make test
```

Runs the center of lane test.

### Video Processing

```bash
make video
```

Processes video input for lane detection.

### Performance Testing

```bash
make fps
```

Runs FPS (Frames Per Second) testing.

### Maintenance

```bash
make freeze
```

Updates requirements.txt with current dependencies.

```bash
make clear-logs
```

Clears all log files.

## Configuration

The project uses `config.json` for configuration settings. Make sure to adjust the parameters according to your needs.

## Test Applications

The project includes several test applications in the `app/` directory:

### TuSimple Dataset Tests

- `tusimple_center_of_lane_test.py`: Tests lane center detection
- `tusimple_center_of_lane_bev_test.py`: Tests lane center detection with Bird's Eye View
- `tusimple_fitted_lanes_test.py`: Tests lane curve fitting
- `tusimple_segmentation_test.py`: Tests lane segmentation
- `tusimple_train_enet.py`: Training script for ENet model
- `tusimple_test_metrics.py`: Evaluation metrics for TuSimple dataset
- `tusimple_enet_benchmark.py`: Benchmark script for ENet model
- `run_all_benchmarks.py`: Runs benchmarks for all models

### Performance and Utility Tests

- `fps_test.py`: Tests frames per second performance
- `camera_calib_test.py`: Tests camera calibration
- `gpu_test.py`: Tests GPU availability and configuration

### Essential Components

- `essentials/`: Contains essential utility modules
- `__init__.py`: Package initialization file

## License

This project is licensed under the terms included in the LICENSE file.
