"""
Logging configuration for the Culture.ai project.
"""
import os
import logging
import logging.handlers
import json
from pathlib import Path

def setup_logging(log_dir="logs"):
    """
    Configure application logging with formatters for different handlers.
    
    Args:
        log_dir: Directory to store log files
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with more concise format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general application logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "app.log", 
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Set up dedicated logger for LLM performance metrics
    llm_perf_logger = logging.getLogger("llm_performance")
    llm_perf_logger.setLevel(logging.INFO)
    
    # Use a separate file handler for LLM performance logs
    llm_file_handler = logging.handlers.RotatingFileHandler(
        log_path / "llm_performance.log",
        maxBytes=50 * 1024 * 1024,  # 50 MB
        backupCount=10
    )
    llm_file_handler.setLevel(logging.INFO)
    
    # Use a simple formatter for the performance logs to make parsing easier
    llm_formatter = logging.Formatter('%(asctime)s - %(message)s')
    llm_file_handler.setFormatter(llm_formatter)
    llm_perf_logger.addHandler(llm_file_handler)
    
    # Make the LLM logger propagate=False to avoid duplicate entries in general log
    llm_perf_logger.propagate = False
    
    # Return the configured loggers
    return root_logger, llm_perf_logger 