# MUS Tuning RAG Assessment Process

This directory contains all the files needed for evaluating the RAG performance of different MUS pruning configurations.

## Step 1: Understand the Assessment Process

Please read the `rag_assessment_instructions.md` file, which provides detailed information about:
- The assessment criteria for RAG responses
- The file structure and locations of responses
- The 3 standard queries used across all scenarios

## Step 2: Evaluate the RAG Responses

1. Fill out the CSV template (`rag_assessment_scores.csv`) with your qualitative assessments.
2. Score each response on 4 criteria (relevance, coherence, completeness, accuracy).
3. Focus on priority scenarios if time is limited (see instructions).

## Step 3: Integrate the Scores

Once you've completed your assessment, run the integration script to update the RAG assessment files:

```
python request_to_operator/integrate_rag_scores.py
```

This will:
- Process your scores from the CSV file
- Update the RAG assessment JSON files with your evaluations
- Calculate average scores for each agent and scenario

## Step 4: Update the Report and Visualizations

After integrating the scores, run the report update script:

```
python request_to_operator/update_report.py
```

This will:
- Update the comparative data with your RAG assessments
- Generate updated visualizations
- Modify the MUS threshold tuning report with your insights
- Provide a final recommendation based on both memory efficiency and RAG performance

## Output

The final, updated report will be available at:
`benchmarks/mus_threshold_tuning_report.md`

## Timeline

Please try to complete this assessment by [INSERT DATE]. Thank you for your valuable input! 