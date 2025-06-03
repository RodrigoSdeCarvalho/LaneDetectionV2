import os
import json
import traceback
from pathlib import Path
from datetime import datetime
import shutil

def run_benchmark(model_path, output_dir):
    """Run the benchmark script for a given model."""
    print(f"\nRunning benchmark for model: {model_path}")
    
    # Import here to avoid circular imports
    from tusimple_enet_benchmark import main as run_benchmark
    
    # Get model name
    model_name = os.path.basename(model_path)
    
    # Run the benchmark with string path and model name
    run_benchmark(str(output_dir), model_name)
    
    # Get the most recent output file from the output directory
    json_files = list(output_dir.glob(f"test_pred-*-{model_name}-*.json"))
    if not json_files:
        raise Exception(f"No output file found for model {model_path}")
    
    # Get the most recent file
    try:
        latest_json = max(json_files, key=os.path.getctime)
    except ValueError:
        raise Exception(f"Could not find any JSON files for model {model_path}")
    
    return latest_json

def run_metrics(pred_file, gt_file):
    """Run the metrics evaluation script."""
    print(f"Running metrics evaluation for: {pred_file}")
    
    # Import here to avoid circular imports
    from app.tusimple_test_metrics import LaneEval
    
    # Run the metrics evaluation
    metrics_json = LaneEval.bench_one_submit(str(pred_file), str(gt_file))
    return json.loads(metrics_json)

def main():
    # Create output directories
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Get all model files
    models_dir = Path("assets/models")
    model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
    
    # Get ground truth file
    gt_file = Path("assets/data/TUSimple/test_set/test_label.json")
    if not gt_file.exists():
        raise Exception(f"Ground truth file not found at {gt_file}")
    
    # Store results for all models
    all_results = {}
    
    # Process each model
    for model_file in model_files:
        try:
            print(f"\nProcessing model: {model_file}")
            
            # Create model-specific output directory
            model_name = model_file.stem
            model_output_dir = outputs_dir / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            # Run benchmark
            pred_file = run_benchmark(model_file, model_output_dir)
            
            # Run metrics
            metrics = run_metrics(pred_file, gt_file)
            
            # Store results
            all_results[model_name] = {
                "metrics": metrics,
                "prediction_file": str(pred_file),
                "model_file": str(model_file)
            }
            
            # Save individual model results
            with open(model_output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Copy prediction file to model directory
            shutil.copy2(pred_file, model_output_dir / "predictions.json")
            
        except Exception as e:
            print(f"Error processing model {model_file}:")
            print("Exception:", str(e))
            print("\nFull stack trace:")
            traceback.print_exc()
            print("\n")
            all_results[model_name] = {
                "error": str(e),
                "stack_trace": traceback.format_exc(),
                "model_file": str(model_file)
            }
    
    # Save summary of all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = outputs_dir / f"benchmark_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBenchmark summary saved to: {summary_file}")

if __name__ == "__main__":
    main() 
