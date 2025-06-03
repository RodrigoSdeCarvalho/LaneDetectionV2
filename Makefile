# Default target
default: help

# Help command
help:
	@echo "Available commands:"
	@echo "  make setup        - Set up the development environment"
	@echo "  make train        - Train the ENet model"
	@echo "  make test         - Run center of lane test"
	@echo "  make video        - Run video processing test"
	@echo "  make fps          - Run FPS performance test"
	@echo "  make benchmark    - Run all benchmarks"	
	@echo "  make freeze       - Update requirements.txt"
	@echo "  make clear-logs   - Clear all log files"
	@echo "  make clean        - Clean up temporary files"

# Setup development environment
setup:
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "Windows: .\venv\Scripts\activate"
	@echo "Linux/Mac: source venv/bin/activate"
	@echo "Then run: pip install -r requirements.txt"

# Training commands
train:
	cd app && python3 train_enet.py

# Testing commands
test:
	cd app && python3 center_of_lane_test.py

# Video processing
video:
	cd app && python3 video_test.py

# Performance testing
fps:
	cd app && python3 fps_test.py

# Benchmarking
benchmark:
	cd app && python3 run_all_benchmarks.py

# Maintenance commands
freeze:
	pipreqs ./

clear-logs:
	rm -rf logs/*
	touch logs/.gitkeep
	clear
	echo "Logs cleared"

# Clean up temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
	clear
	echo "Cleaned up temporary files"
