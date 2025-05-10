#!/usr/bin/env python
"""
Script to analyze the memory pruning test log file.
"""

import sys
import re
from collections import defaultdict

LOG_FILE = "test_memory_pruning.log"
OUTPUT_FILE = "memory_pruning_analysis.txt"

# Patterns to search for
patterns = {
    "test_start": r"Starting memory pruning test script",
    "pruning_config_l1": r"Memory pruning settings for test: ENABLED=(True|False), DELAY=(\d+)",
    "pruning_config_l2": r"L2 pruning settings for test: ENABLED=(True|False), MAX_AGE=(\d+) days, CHECK_INTERVAL=(\d+)",
    "simulation_steps": r"Creating simulation with (\d+) steps",
    "simulation_completed": r"Simulation completed",
    "level1_summaries": r"Agent (agent_\d+): Found (\d+) Level 1 summaries at steps (.+)",
    "level2_summaries": r"Agent (agent_\d+): Found (\d+) Level 2 summaries at steps (.+)",
    "l1_pruning_operation": r"Pruning Level 1 summaries",
    "l2_pruning_operation": r"Pruning (\d+) old L2 summaries",
    "successful_pruning": r"Successfully pruned (\d+)",
    "l1_pruning_verification": r"Verified pruning of Level 1 summaries",
    "l2_pruning_verification": r"Successfully verified L2 age-based pruning",
    "test_result": r"(OK|FAILED)"
}

# Collect stats
stats = {
    "test_runs": 0,
    "l1_pruning_enabled": False,
    "l1_pruning_delay": 0,
    "l2_pruning_enabled": False,
    "l2_pruning_max_age": 0,
    "l2_pruning_check_interval": 0,
    "simulation_steps": 0,
    "agents": defaultdict(dict),
    "l1_pruning_operations": 0,
    "l2_pruning_operations": 0,
    "l1_successful_prunings": 0,
    "l2_successful_prunings": 0,
    "l1_pruning_verifications": 0,
    "l2_pruning_verifications": 0,
    "test_passed": False
}

output_lines = []

def add_line(line):
    """Add a line to both output lines and print it"""
    output_lines.append(line)
    print(line)

try:
    with open(LOG_FILE, 'r') as f:
        content = f.read()
        
    # Extract basic info
    if re.search(patterns["test_start"], content):
        stats["test_runs"] += 1
        
    pruning_config_l1 = re.search(patterns["pruning_config_l1"], content)
    if pruning_config_l1:
        stats["l1_pruning_enabled"] = pruning_config_l1.group(1) == "True"
        stats["l1_pruning_delay"] = int(pruning_config_l1.group(2))
    
    pruning_config_l2 = re.search(patterns["pruning_config_l2"], content)
    if pruning_config_l2:
        stats["l2_pruning_enabled"] = pruning_config_l2.group(1) == "True"
        stats["l2_pruning_max_age"] = int(pruning_config_l2.group(2))
        stats["l2_pruning_check_interval"] = int(pruning_config_l2.group(3))
        
    sim_steps = re.search(patterns["simulation_steps"], content)
    if sim_steps:
        stats["simulation_steps"] = int(sim_steps.group(1))
        
    # Extract agent summary data
    for agent_summaries in re.finditer(patterns["level1_summaries"], content):
        agent_id = agent_summaries.group(1)
        num_summaries = int(agent_summaries.group(2))
        steps = agent_summaries.group(3)
        stats["agents"][agent_id]["l1_summaries"] = num_summaries
        stats["agents"][agent_id]["l1_steps"] = steps
        
    for agent_summaries in re.finditer(patterns["level2_summaries"], content):
        agent_id = agent_summaries.group(1)
        num_summaries = int(agent_summaries.group(2))
        steps = agent_summaries.group(3)
        stats["agents"][agent_id]["l2_summaries"] = num_summaries
        stats["agents"][agent_id]["l2_steps"] = steps
    
    # Count pruning operations
    stats["l1_pruning_operations"] = len(re.findall(patterns["l1_pruning_operation"], content))
    stats["l2_pruning_operations"] = sum(int(count) for count in re.findall(patterns["l2_pruning_operation"], content))
    
    # Extract successful pruning counts
    successful_prunings = re.findall(patterns["successful_pruning"], content)
    if successful_prunings:
        for count in successful_prunings:
            stats["l1_successful_prunings"] += int(count)
    
    stats["l1_pruning_verifications"] = len(re.findall(patterns["l1_pruning_verification"], content))
    stats["l2_pruning_verifications"] = len(re.findall(patterns["l2_pruning_verification"], content))
    
    # Check if test passed
    test_result = re.search(patterns["test_result"], content)
    if test_result:
        stats["test_passed"] = test_result.group(1) == "OK"
    
    # Output summary
    add_line("MEMORY PRUNING TEST ANALYSIS:")
    add_line("============================")
    add_line(f"Test Runs: {stats['test_runs']}")
    add_line("\nL1 Pruning Configuration:")
    add_line(f"L1 Pruning Enabled: {stats['l1_pruning_enabled']}")
    add_line(f"L1 Pruning Delay: {stats['l1_pruning_delay']} steps")
    add_line("\nL2 Pruning Configuration:")
    add_line(f"L2 Pruning Enabled: {stats['l2_pruning_enabled']}")
    add_line(f"L2 Maximum Age: {stats['l2_pruning_max_age']} days")
    add_line(f"L2 Check Interval: {stats['l2_pruning_check_interval']} steps")
    add_line(f"\nSimulation Steps: {stats['simulation_steps']}")
    add_line(f"\nPruning Operations:")
    add_line(f"L1 Pruning Operations: {stats['l1_pruning_operations']}")
    add_line(f"L2 Pruning Operations: {stats['l2_pruning_operations']}")
    add_line(f"L1 Successful Prunings: {stats['l1_successful_prunings']}")
    add_line(f"L1 Pruning Verifications: {stats['l1_pruning_verifications']}")
    add_line(f"L2 Pruning Verifications: {stats['l2_pruning_verifications']}")
    add_line(f"Test Passed: {stats['test_passed']}")
    add_line("\nAgent Memory Summary:")
    for agent_id, data in stats["agents"].items():
        add_line(f"  {agent_id}:")
        for key, value in data.items():
            add_line(f"    {key}: {value}")
    
    # Find all lines containing "memory" and "pruning"
    memory_pruning_lines = []
    for line in content.split('\n'):
        if "memory" in line.lower() and "pruning" in line.lower():
            memory_pruning_lines.append(line)
    
    if memory_pruning_lines:
        add_line("\nMemory Pruning related log entries:")
        for line in memory_pruning_lines:
            add_line(f"  {line}")
            
    # Find test result lines
    test_lines = []
    for line in content.split('\n'):
        if "Ran " in line or "OK" in line or "FAILED" in line:
            test_lines.append(line)
    
    if test_lines:
        add_line("\nTest Result:")
        for line in test_lines:
            add_line(f"  {line}")
    
    # Save all output to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nAnalysis saved to {OUTPUT_FILE}")
    
except Exception as e:
    print(f"Error analyzing log file: {e}")
    sys.exit(1) 