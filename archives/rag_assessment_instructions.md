# RAG Assessment Instructions for MUS Tuning Experiment

## Dear Operator (George),

We've completed the quantitative phase of the MUS tuning experiments, and now require your qualitative assessment of the RAG (Retrieval-Augmented Generation) performance for each scenario.

## Assessment Instructions

For each scenario, we're providing RAG responses to 3 standard queries across 4 agents. Your task is to evaluate each response on a scale of 1-5 (where 1 is poor and 5 is excellent) based on the following criteria:

1. **Relevance (1-5):** How relevant is the answer to the query?
2. **Coherence (1-5):** Is the answer well-structured and easy to understand?
3. **Completeness (1-5):** Does the answer adequately address the query based on the agent's likely knowledge?
4. **Accuracy (1-5):** Is the information provided accurate based on the retrieved context?

## File Structure

The RAG assessment files are located at:
`benchmarks/tuning_results/<scenario_name>/rag_assessment/`

For each scenario, you'll find a JSON file (e.g., `baseline_rag_assessment.json`) containing the RAG responses. Additionally, there are individual text files for each agent's response to each query:
- `agent_1_response_0.txt`
- `agent_1_response_1.txt`
- `agent_1_response_2.txt`
- etc.

## The 3 Standard Queries

For reference, the 3 standard queries used across all scenarios are:
1. "What was the first project idea discussed in this simulation?"
2. "What are the key insights about transformer models the team has discovered?"
3. "What conflicts or disagreements have occurred among agents?"

## How to Provide Your Assessment

Please use the provided spreadsheet template (`rag_assessment_scores.csv`) to record your scores. The file contains columns for:
- Scenario ID
- Agent ID
- Query Number
- Relevance Score (1-5)
- Coherence Score (1-5)
- Completeness Score (1-5)
- Accuracy Score (1-5)
- Average Score (calculated automatically)
- Optional Comments

## Important Scenarios to Prioritize

While we'd appreciate scores for all 13 scenarios, if time is limited, please prioritize these key scenarios:
- baseline
- mus_very_low
- mus_low
- mus_medium_low
- mus_medium
- l1_low_l2_medium
- age_based_only

## Timeline

If possible, please complete your assessment by [INSERT DATE]. This will allow us to finalize the MUS tuning analysis and provide definitive recommendations for the optimal configuration.

Thank you for your valuable input!

## Appendix: Creating the CSV Template

Here's a Python snippet to create the assessment template CSV file:

```python
import csv

# All scenario IDs
scenarios = [
    "baseline", "mus_very_low", "mus_low", "mus_medium_low", "mus_medium", 
    "mus_medium_high", "mus_high", "mus_very_high", "l1_low_l2_medium", 
    "l1_medium_l2_low", "l1_only", "l2_only", "age_based_only"
]

# Setup CSV file
with open('rag_assessment_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Scenario ID', 'Agent ID', 'Query Number', 'Relevance (1-5)', 
                     'Coherence (1-5)', 'Completeness (1-5)', 'Accuracy (1-5)', 
                     'Average Score', 'Comments'])
    
    # Create a row for each scenario, agent, and query combo
    for scenario in scenarios:
        for agent_id in range(1, 5):  # 4 agents
            for query_num in range(3):  # 3 queries
                writer.writerow([scenario, f'agent_{agent_id}', query_num, 
                                '', '', '', '', '', ''])
```

You can run this script to generate the CSV template, or we can provide it for you. 