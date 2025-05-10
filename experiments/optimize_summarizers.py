#!/usr/bin/env python
"""
Combined script to run both L1 and L2 summarizer optimizations.

This script:
1. Runs the L1 summarizer optimization
2. Runs the L2 summarizer optimization
3. Generates a combined report of the results
"""

import sys
import os
import logging
import json
import subprocess
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiments/summarizer_optimization.log")
    ]
)
logger = logging.getLogger("summarizer_optimizer")

def run_optimization(script_name):
    """
    Run an optimization script and capture its output.
    
    Args:
        script_name: Name of the script to run
        
    Returns:
        bool: True if the optimization was successful, False otherwise
    """
    script_path = os.path.join(project_root, "experiments", script_name)
    
    if not os.path.exists(script_path):
        logger.error(f"Script {script_path} does not exist")
        return False
    
    logger.info(f"Running {script_name}...")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        logger.info(f"{script_name} execution complete")
        logger.debug(f"Output: {result.stdout}")
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"{script_name} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {e}")
        return False

def generate_combined_report():
    """
    Generate a combined report of the L1 and L2 optimization results.
    
    Returns:
        bool: True if report generation was successful, False otherwise
    """
    try:
        # Check for the individual report files
        l1_report_path = os.path.join(project_root, "experiments", "l1_summarizer_optimization_report.json")
        l2_report_path = os.path.join(project_root, "experiments", "l2_summarizer_optimization_report.json")
        
        l1_report = None
        l2_report = None
        
        if os.path.exists(l1_report_path):
            with open(l1_report_path, 'r') as f:
                l1_report = json.load(f)
                logger.info("Loaded L1 optimization report")
        
        if os.path.exists(l2_report_path):
            with open(l2_report_path, 'r') as f:
                l2_report = json.load(f)
                logger.info("Loaded L2 optimization report")
        
        # Create the combined report
        combined_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "l1_optimization": l1_report if l1_report else "Not available",
            "l2_optimization": l2_report if l2_report else "Not available",
            "summary": {
                "l1_improvement": l1_report["improvement"] if l1_report else "N/A",
                "l2_improvement": l2_report["improvement"] if l2_report else "N/A"
            }
        }
        
        # Add overall success metrics
        combined_report["summary"]["l1_successful"] = (l1_report is not None)
        combined_report["summary"]["l2_successful"] = (l2_report is not None)
        combined_report["summary"]["overall_successful"] = (l1_report is not None and l2_report is not None)
        
        # Save the combined report
        combined_report_path = os.path.join(project_root, "experiments", "dspy_summarizers_optimization_report.json")
        with open(combined_report_path, 'w') as f:
            json.dump(combined_report, f, indent=2)
        
        logger.info(f"Combined report saved to {combined_report_path}")
        
        # Generate a markdown report for better readability
        markdown_report = f"""# DSPy Summarizers Optimization Report

**Timestamp:** {combined_report['timestamp']}

## Overview

This report summarizes the results of optimizing the L1 and L2 summarizers using DSPy BootstrapFewShot teleprompters.

| Summarizer | Base Score | Optimized Score | Improvement | Success |
|------------|------------|-----------------|-------------|---------|
| L1 | {l1_report['base_model_score'] if l1_report else 'N/A'} | {l1_report['optimized_model_score'] if l1_report else 'N/A'} | {l1_report['improvement'] if l1_report else 'N/A'} | {'Yes' if l1_report else 'No'} |
| L2 | {l2_report['base_model_score'] if l2_report else 'N/A'} | {l2_report['optimized_model_score'] if l2_report else 'N/A'} | {l2_report['improvement'] if l2_report else 'N/A'} | {'Yes' if l2_report else 'No'} |

## Details

### L1 Summarizer Optimization

{"Successfully optimized." if l1_report else "Optimization failed or did not improve performance."}

{f"- Base Model Score: {l1_report['base_model_score']}" if l1_report else ""}
{f"- Optimized Model Score: {l1_report['optimized_model_score']}" if l1_report else ""}
{f"- Improvement: {l1_report['improvement']} ({l1_report['percent_improvement']}%)" if l1_report else ""}
{f"- Model: {l1_report['model']}" if l1_report else ""}
{f"- Temperature: {l1_report['temperature']}" if l1_report else ""}
{f"- Method: {l1_report['optimization_method']}" if l1_report else ""}

### L2 Summarizer Optimization

{"Successfully optimized." if l2_report else "Optimization failed or did not improve performance."}

{f"- Base Model Score: {l2_report['base_model_score']}" if l2_report else ""}
{f"- Optimized Model Score: {l2_report['optimized_model_score']}" if l2_report else ""}
{f"- Improvement: {l2_report['improvement']} ({l2_report['percent_improvement']}%)" if l2_report else ""}
{f"- Model: {l2_report['model']}" if l2_report else ""}
{f"- Temperature: {l2_report['temperature']}" if l2_report else ""}
{f"- Method: {l2_report['optimization_method']}" if l2_report else ""}

## Conclusion

The optimization process {'was successful for both L1 and L2 summarizers' if combined_report['summary']['overall_successful'] else 'was partially successful' if combined_report['summary']['l1_successful'] or combined_report['summary']['l2_successful'] else 'failed'}.
The optimized models {'have been' if combined_report['summary']['l1_successful'] or combined_report['summary']['l2_successful'] else 'were not'} saved to the compiled directory and can be used by the agent's memory system.
"""
        
        # Save the markdown report
        markdown_report_path = os.path.join(project_root, "experiments", "dspy_summarizers_optimization_report.md")
        with open(markdown_report_path, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Markdown report saved to {markdown_report_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating combined report: {e}")
        return False

def main():
    """
    Main function to run the optimization process.
    """
    logger.info("Starting DSPy summarizers optimization")
    
    # Run L1 optimization
    l1_success = run_optimization("optimize_l1_summarizer.py")
    logger.info(f"L1 summarizer optimization {'succeeded' if l1_success else 'failed'}")
    
    # Run L2 optimization
    l2_success = run_optimization("optimize_l2_summarizer.py")
    logger.info(f"L2 summarizer optimization {'succeeded' if l2_success else 'failed'}")
    
    # Generate combined report
    report_success = generate_combined_report()
    logger.info(f"Combined report generation {'succeeded' if report_success else 'failed'}")
    
    # Overall success
    overall_success = l1_success and l2_success and report_success
    
    if overall_success:
        logger.info("DSPy summarizers optimization completed successfully")
        return 0
    else:
        logger.error("DSPy summarizers optimization had issues")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 