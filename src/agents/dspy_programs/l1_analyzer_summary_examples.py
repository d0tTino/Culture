"""
DSPy L1 Analyzer Summary Examples

This module provides example data for training and testing DSPy-based L1 summary generation
specifically tailored for agents in the Analyzer role.
"""

import dspy

# Example 1: Analyzer evaluating project progress with metrics
example1 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 23, Thought: I should review our progress on the NLP model and identify potential improvements.
    - Step 23, Broadcast Message: "I've been analyzing our NLP model's performance metrics from the last three test runs. While accuracy has improved by 4.2%, we're seeing diminishing returns from additional training data."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 24, Perceived Message: Agent_1 asked: "What do you think is causing the diminishing returns?"
    - Step 24, Thought: This is likely due to model architecture limitations rather than data quality issues.
    - Step 24, Targeted Message: Sent to Agent_1: "Based on my analysis, it appears to be an architectural limitation. Our current attention mechanism doesn't effectively capture long-range dependencies in text. We should consider experimenting with alternative architectures like Transformer-XL or Reformer."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 25, Perceived Message: Agent_5 said: "I've implemented Transformer-XL in previous projects. The computational requirements are significantly higher, but the results justify it for complex language tasks."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 25, Thought: That's useful information. We should weigh the trade-offs carefully.
    - Step 25, Broadcast Message: "Thank you for sharing that experience, Agent_5. Let's quantify these trade-offs more precisely. I'll prepare a cost-benefit analysis comparing our current architecture with Transformer-XL and Reformer, considering both performance improvements and computational requirements."  # noqa: E501  # Long test string; further breaking harms readability.
    """,
    current_mood="focused",
    l1_summary=(
        "I conducted a detailed analysis of our NLP model's performance metrics across multiple test runs, "
        "identifying a precise 4.2% accuracy improvement alongside diminishing returns from additional training data. "
        "My systematic evaluation determined this was likely due to architectural limitations in capturing long-range text dependencies rather than data quality issues. "  # Long test string; further breaking harms readability.
        "I recommended specific alternative architectures (Transformer-XL or Reformer) based on technical capabilities rather than general improvements. "
        "When Agent_5 provided qualitative feedback about Transformer-XL's computational requirements, I immediately proposed quantifying these trade-offs through a rigorous cost-benefit analysis comparing performance gains against computational costs across multiple architectures. "  # Long test string; further breaking harms readability.
        "Throughout this exchange, I maintained a focused analytical approach, prioritizing empirical evidence and quantitative assessment to drive decision-making about technical architecture choices."  # Long test string; further breaking harms readability.
    ),
)

# Example 2: Analyzer identifying data inconsistencies
example2 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 17, Thought: I've noticed some unusual patterns while reviewing the user acquisition data from last month.
    - Step 17, Broadcast Message: "Upon examining the user acquisition funnel data, I've identified a statistical anomaly. Conversion rates from step 3 to 4 dropped by 27% on mobile devices, but only during weekday afternoons. This pattern doesn't appear on weekends or other times of day."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 18, Perceived Message: Agent_2 said: "That's strange. Could it be related to the A/B test we were running during that period?"
    - Step 18, Thought: I need to cross-reference this with the A/B test data to check for correlation.
    - Step 18, Broadcast Message: "Good hypothesis. I've cross-referenced with the A/B test schedules and found that variant B was indeed being served to 50% of mobile users during that period. However, analyzing the segmented data shows the conversion drop affected both test and control groups equally, suggesting the A/B test is not the cause."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 19, Perceived Message: Agent_4 asked: "Could there be external factors? Maybe network issues or competing work activities during those specific times?"
    - Step 19, Thought: The timing pattern is highly suggestive of an environmental factor. Let me check server and network metrics.
    - Step 19, Broadcast Message: "Excellent insight. After examining server response times and network logs, I've found a correlation with increased API latency specifically on weekday afternoons. Mobile connections appear most affected, with response times averaging 2.3x longer than baseline during these periods, exceeding our 1.5-second threshold for optimal user experience."  # noqa: E501  # Long test string; further breaking harms readability.
    """,
    current_mood="methodical",
    l1_summary=(
        "I identified a precise statistical anomaly in user acquisition data: a 27% drop in conversion rates from step 3 to 4 "
        "specifically on mobile devices during weekday afternoons, with no similar pattern on weekends or other times. "
        "When Agent_2 suggested a connection to ongoing A/B testing, I methodically cross-referenced with test schedules and performed segmented data analysis, "  # Long test string; further breaking harms readability.
        "conclusively determining that both test and control groups were equally affected, eliminating the A/B test as a causal factor. "
        "Agent_4's hypothesis about external factors prompted me to analyze server and network metrics, where I discovered a direct correlation with increased API latency during the specific time windows—mobile connections experienced precisely 2.3x longer response times, exceeding our established 1.5-second threshold for optimal user experience. "  # Long test string; further breaking harms readability.
        "My analysis systematically narrowed down potential causes through evidence-based elimination and precise quantification, ultimately identifying the technical root cause of the conversion anomaly."  # Long test string; further breaking harms readability.
    ),
)

# Example 3: Analyzer identifying logical contradiction in project requirements
example3 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 29, Perceived Message: Agent_1 said: "For the new recommendation system, we need to ensure both high personalization and complete user privacy. Let's aim for maximum customization without storing any user data."
    - Step 29, Thought: This requirement contains a fundamental logical contradiction that needs to be addressed.
    - Step 29, Broadcast Message: "I've identified a critical contradiction in our requirements. Personalization fundamentally requires some form of user data or state persistence. We cannot simultaneously maximize personalization and eliminate all user data storage—these goals exist on a spectrum where improving one necessitates compromising the other."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 30, Perceived Message: Agent_3 said: "Maybe we could use local storage on the user's device instead of our servers?"
    - Step 30, Thought: That's a potential compromise worth analyzing.
    - Step 30, Broadcast Message: "Local storage offers a partial solution but introduces three measurable trade-offs: 1) Personalization would be device-specific, reducing effectiveness for multi-device users by approximately 40% based on our user data; 2) We lose cross-user collaborative filtering capabilities, which currently drive 63% of our successful recommendations; and 3) Cold-start problems would be exacerbated, increasing new user onboarding friction by an estimated 35%."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 31, Perceived Message: Agent_1 said: "Those are significant concerns. What if we use federated learning instead?"
    - Step 31, Thought: Federated learning presents its own set of trade-offs that need careful analysis.
    - Step 31, Broadcast Message: "Federated learning could indeed balance these requirements more effectively. Based on published benchmarks and our user patterns, I estimate we could retain approximately 85% of personalization effectiveness while addressing the main privacy concerns. However, this would increase computational requirements by 2.7x and introduce model convergence challenges in our highly heterogeneous user base. I recommend a small-scale technical proof of concept to validate these projections."  # noqa: E501  # Long test string; further breaking harms readability.
    """,
    current_mood="analytical",
    l1_summary=(
        "I identified a fundamental logical contradiction in our project requirements: the mutual exclusivity between maximizing personalization and eliminating user data storage. "
        "When Agent_3 suggested using local storage as a solution, I conducted a three-point quantitative analysis of specific trade-offs: 1) 40% reduced effectiveness for multi-device users based on our user patterns, 2) loss of collaborative filtering which drives 63% of successful recommendations, and 3) approximately 35% increased friction in new user onboarding due to cold-start problems. "  # Long test string; further breaking harms readability.
        "When Agent_1 proposed federated learning, I provided a data-backed assessment estimating 85% personalization effectiveness retention while addressing privacy concerns, but noted the precise 2.7x increase in computational requirements and convergence challenges in our heterogeneous user base. "  # Long test string; further breaking harms readability.
        "My analytical approach transformed a contradictory requirement into a structured evaluation of implementation options with quantified trade-offs, culminating in a recommendation for an evidence-based proof of concept to validate projections."  # Long test string; further breaking harms readability.
    ),
)

# Example 4: Analyzer evaluating competing algorithms
example4 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 36, Thought: We need to select the most appropriate clustering algorithm for our user segmentation. I should evaluate the options systematically.
    - Step 36, Broadcast Message: "I've compared five clustering algorithms (K-means, DBSCAN, Hierarchical, Gaussian Mixture, and BIRCH) for our user segmentation task using three key metrics. Testing with our dataset yielded the following: 1) Silhouette scores: K-means 0.68, DBSCAN 0.72, Hierarchical 0.65, GMM 0.71, BIRCH 0.67; 2) Execution time: K-means 12s, DBSCAN 45s, Hierarchical 78s, GMM 34s, BIRCH 15s; 3) Memory usage: K-means 1.2GB, DBSCAN 2.1GB, Hierarchical 2.8GB, GMM 1.8GB, BIRCH 1.3GB."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 37, Perceived Message: Agent_2 asked: "What about interpretability of the clusters? That's important for our marketing team."
    - Step 37, Thought: That's a valid point that wasn't included in my initial analysis. I should evaluate this aspect as well.
    - Step 37, Broadcast Message: "Excellent point about interpretability. After reviewing the cluster structures: K-means and BIRCH produced the most distinct and interpretable segments (rated 8/10 and 7/10 respectively in our marketing team's blind evaluation), while DBSCAN created more irregular shapes that were harder to describe (5/10). GMM provided probabilistic assignments that marketing found useful but complex (6/10). Hierarchical clustering's dendrogram offered valuable insights into relationship structure but was considered too technical for routine use (4/10)."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 38, Perceived Message: Agent_5 asked: "What about stability across different data samples? We'll be rerunning this segmentation monthly."
    - Step 38, Thought: Stability is indeed crucial for longitudinal analysis. I should evaluate this aspect.
    - Step 38, Broadcast Message: "I've assessed algorithm stability by running each on 10 different bootstrapped samples from our data. Measuring cluster assignment consistency: K-means showed moderate variability (78% consistency), DBSCAN was more stable (85% consistency), Hierarchical was highly stable (92% consistency), GMM showed lower stability (71% consistency), while BIRCH achieved 82% consistency. For monthly reanalysis, the stability metrics suggest hierarchical clustering would provide the most consistent segments over time, despite its longer execution time."  # noqa: E501  # Long test string; further breaking harms readability.
    """,
    current_mood="thorough",
    l1_summary=(
        "I conducted a systematic comparison of five clustering algorithms (K-means, DBSCAN, Hierarchical, Gaussian Mixture, and BIRCH) for user segmentation using multiple quantitative metrics. "
        "My initial analysis provided precise values for silhouette scores (ranging from 0.65 to 0.72), execution times (12s to 78s), and memory usage (1.2GB to 2.8GB) for each algorithm. "  # Long test string; further breaking harms readability.
        "When Agent_2 inquired about interpretability, I extended my analysis to include structured marketing team evaluations, quantifying interpretability from 4/10 (Hierarchical) to 8/10 (K-means). "  # Long test string; further breaking harms readability.
        "Agent_5's question about stability prompted further empirical testing using bootstrapped sampling, revealing specific consistency percentages ranging from 71% (GMM) to 92% (Hierarchical). "  # Long test string; further breaking harms readability.
        "My thorough, multi-dimensional analysis evolved beyond simplistic recommendations to present a comprehensive trade-off framework, ultimately suggesting that despite longer execution time, Hierarchical clustering's superior stability (92% consistency) makes it optimal for our recurring monthly segmentation needs. "  # Long test string; further breaking harms readability.
        "This demonstrates my analytical approach of systematically evaluating multiple solutions across diverse, relevant dimensions before reaching evidence-based conclusions."
    ),
)

# Example 5: Analyzer debugging performance regression
example5 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 41, Perceived Message: Agent_4 said: "Our application response time has increased by almost 40% since the last deployment. Users are complaining about slowness."
    - Step 41, Thought: I need to systematically isolate the cause of this performance regression.
    - Step 41, Broadcast Message: "I'll analyze this performance regression through systematic investigation. First, let me establish if this is a frontend or backend issue by comparing server-side vs. client-side timing metrics."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 42, Thought: After reviewing the timing data, I can see the issue is primarily on the backend.
    - Step 42, Broadcast Message: "Analysis of timing metrics shows this is predominantly a backend issue. Server response times increased by 37.8%, while frontend rendering times only increased by 3.2%. I've identified three potential causes based on the correlation with our deployment: 1) The new caching layer implementation, 2) Updated database query patterns, or 3) Changes to the authentication flow."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 43, Thought: Let me isolate each component to identify the specific bottleneck.
    - Step 43, Broadcast Message: "I've isolated each component using controlled testing. Results show the performance regression is specifically tied to the new database query patterns. The N+1 query problem has been introduced in the user preference lookup functionality—each user request now triggers 12 separate database queries instead of the previous batched approach. This creates a multiplicative effect during peak usage."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 44, Perceived Message: Agent_2 asked: "Can we revert just that part of the changes rather than the entire deployment?"
    - Step 44, Thought: Let me check if there's a viable selective rollback approach.
    - Step 44, Broadcast Message: "I've analyzed the deployment structure and identified that we can selectively revert the query pattern changes without affecting other improvements. The specific issue is isolated to the UserPreferenceService class, methods fetchUserSettings() and applyUserPreferences(). Reverting these methods to version 2.3.0 while maintaining other updates would resolve the performance regression with minimal impact. I've prepared a targeted patch that implements this fix."  # noqa: E501  # Long test string; further breaking harms readability.
    """,
    current_mood="systematic",
    l1_summary=(
        "When faced with reports of a 40% application response time increase after deployment, I implemented a systematic debugging methodology to isolate the cause. "
        "First, I differentiated between frontend and backend issues by comparing timing metrics, precisely quantifying a 37.8% increase in server response times versus only 3.2% in frontend rendering—clearly localizing the problem on the backend. "  # Long test string; further breaking harms readability.
        "I then formulated three specific hypotheses based on deployment changes: the new caching layer, updated database query patterns, or authentication flow modifications. "
        "Through controlled component isolation testing, I definitively identified the exact cause: an N+1 query problem in the user preference lookup, where each request now triggered exactly 12 separate database queries instead of using the previous batched approach. "  # Long test string; further breaking harms readability.
        "When Agent_2 asked about selective reversion, I analyzed the deployment structure and pinpointed the specific code locations (UserPreferenceService class, methods fetchUserSettings() and applyUserPreferences()), enabling a targeted fix without sacrificing other improvements. "  # Long test string; further breaking harms readability.
        "My systematic approach enabled rapid, precise problem identification without resorting to blanket solutions or guesswork."
    ),
)

# Example 6: Analyzer evaluating experiment results
example6 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 55, Thought: The A/B test for the new onboarding flow has completed. I should analyze the results comprehensively.
    - Step 55, Broadcast Message: "I've analyzed the results from our 2-week A/B test of the new onboarding flow. Based on data from 24,362 users (12,188 control, 12,174 variant), the new flow demonstrated a statistically significant improvement in completion rate (67.3% vs 58.9%, p<0.001) and reduced average completion time (3.2 minutes vs 4.7 minutes). However, the new flow showed no significant difference in 30-day retention (42.1% vs 41.8%, p=0.74)."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 56, Perceived Message: Agent_1 asked: "Did the improvements apply equally across user segments?"
    - Step 56, Thought: I should analyze segment-specific performance to identify any variations.
    - Step 56, Broadcast Message: "Segment analysis reveals important variations. The completion rate improvement was significantly higher for mobile users (+12.7 percentage points, p<0.001) compared to desktop users (+4.8 percentage points, p=0.03). Additionally, new flow benefits were greater for users in the 18-34 age demographic (+11.2 percentage points) than those 35+ (+5.3 percentage points). Interestingly, users from English-speaking countries showed less improvement (+5.9 percentage points) than non-English users (+14.6 percentage points), suggesting the visual-focused approach better serves our international audience."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 57, Perceived Message: Agent_3 asked: "How did the new flow affect conversion to paid subscriptions?"
    - Step 57, Thought: This is an important business metric that I should have included in my initial analysis.
    - Step 57, Broadcast Message: "Looking at 14-day conversion to paid subscription rates, the new flow showed a modest but statistically significant improvement: 8.4% vs 7.2% in the control group (p=0.02), representing a relative increase of 16.7%. However, the average revenue per user showed no significant difference ($5.23 vs $5.18, p=0.68). When analyzed by tier, the new flow primarily increased conversions to our basic tier (+1.5 percentage points) with no significant impact on premium tier adoption (-0.3 percentage points, p=0.41)."  # noqa: E501  # Long test string; further breaking harms readability.
    """,
    current_mood="meticulous",
    l1_summary=(
        "I analyzed comprehensive results from our 2-week A/B test of the new onboarding flow, using a statistically robust sample of 24,362 users split nearly equally between control and variant groups. "
        "My analysis identified statistically significant improvements in completion rate (67.3% vs 58.9%, p<0.001) and completion time (3.2 vs 4.7 minutes), while finding no significant impact on 30-day retention (p=0.74). "  # Long test string; further breaking harms readability.
        "When Agent_1 requested segment analysis, I conducted a multi-dimensional statistical breakdown, revealing precisely quantified, statistically significant variations across platforms (mobile users showing +12.7 percentage points improvement vs +4.8 for desktop), age demographics (18-34 age group showing +11.2 percentage points vs +5.3 for 35+), and language segments (non-English users benefiting substantially more at +14.6 percentage points vs +5.9). "  # Long test string; further breaking harms readability.
        "Agent_3's question about business impact prompted further analysis of conversion metrics, where I identified a statistically significant 16.7% relative increase in paid subscription conversion (8.4% vs 7.2%, p=0.02), while noting no significant ARPU difference (p=0.68) and a tier-specific pattern of increased basic tier adoption without premium tier impact. "  # Long test string; further breaking harms readability.
        "My meticulous approach ensured comprehensive statistical validation across multiple dimensions and metrics, providing actionable insights about differential segment impacts."  # Long test string; further breaking harms readability.
    ),
)

# Example 7: Analyzer evaluating learning algorithm improvements
example7 = dspy.Example(
    agent_role="Analyzer",
    recent_events="""
    - Step 62, Thought: We need to evaluate whether the gradient boosting or neural network approach performs better for our prediction task. I should design a rigorous comparison.
    - Step 62, Broadcast Message: "I've designed a cross-validation evaluation framework to compare our gradient boosting and neural network approaches for the prediction task. The evaluation will use 5-fold cross-validation with stratification to ensure consistent class distribution across folds, and will assess models on accuracy, F1-score, AUC-ROC, and inference time."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 63, Thought: The evaluation has completed. I should analyze the results.
    - Step 63, Broadcast Message: "The cross-validation results show meaningful differences between approaches. The gradient boosting model achieved mean accuracy of 84.2% (±1.3%), F1-score of 0.79 (±0.02), and AUC-ROC of 0.87 (±0.01). The neural network achieved accuracy of 85.7% (±1.6%), F1-score of 0.81 (±0.02), and AUC-ROC of 0.89 (±0.02). While the neural network shows slightly better performance metrics, its inference time is 215ms per example versus 42ms for gradient boosting. Additionally, gradient boosting provides native feature importance metrics, revealing that features X, Y, and Z contribute 68% of total predictive power."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 64, Perceived Message: Agent_5 asked: "How robust are these models to missing data, which is common in our production environment?"
    - Step 64, Thought: This is an important practical consideration I should evaluate.
    - Step 64, Broadcast Message: "I conducted additional robustness testing by introducing synthetic missing data at rates of 5%, 10%, and 20%. The gradient boosting model showed minimal performance degradation at 5% missing data (accuracy drop of 0.4%), moderate impact at 10% (1.7% drop), and significant impact at 20% (4.3% drop). The neural network was more sensitive, with accuracy drops of 1.2%, 3.8%, and 7.9% respectively. Both models performed worse on the 'transactional' data subset compared to 'demographic' data, suggesting we should prioritize completeness of transactional features."  # noqa: E501  # Long test string; further breaking harms readability.
    - Step 65, Perceived Message: Agent_2 asked: "Given these results, which approach would you recommend we implement?"
    - Step 65, Thought: I need to provide a recommendation based on a holistic evaluation of the trade-offs.
    - Step 65, Broadcast Message: "Based on the comprehensive evaluation, I recommend implementing the gradient boosting model for the initial deployment for three quantifiable reasons: 1) Its inference time is 5.1x faster, which is critical for our real-time application with SLA requirements; 2) It demonstrates superior robustness to missing data, which occurs in approximately 8% of production cases based on current metrics; and 3) It provides interpretable feature importance, which allows for ongoing refinement. While the neural network achieved marginally better accuracy (+1.5%) and AUC-ROC (+0.02), these advantages don't outweigh the operational benefits of gradient boosting for our specific use case and constraints."  # noqa: E501  # Long test string; further breaking harms readability.
    """,
    current_mood="precise",
    l1_summary=(
        "I designed and executed a rigorous comparison between gradient boosting and neural network approaches using 5-fold stratified cross-validation across multiple performance dimensions. "
        "My analysis quantified precise performance metrics with confidence intervals: the neural network achieved slightly better predictive metrics (accuracy: 85.7% ±1.6%, F1: 0.81 ±0.02, AUC-ROC: 0.89 ±0.02) versus gradient boosting (accuracy: 84.2% ±1.3%, F1: 0.79 ±0.02, AUC-ROC: 0.87 ±0.01). "  # Long test string; further breaking harms readability.
        "However, I also evaluated practical considerations, measuring the neural network's 5.1x slower inference time (215ms vs 42ms) and extracting feature importance insights showing features X, Y, and Z contributing 68% of predictive power in the gradient boosting model. "  # Long test string; further breaking harms readability.
        "Agent_5's question prompted additional robustness testing with synthetic missing data at three specific rates, revealing gradient boosting's superior resilience (0.4%, 1.7%, and 4.3% accuracy drops) compared to the neural network's higher sensitivity (1.2%, 3.8%, and 7.9% drops), with particular vulnerability in transactional data. "  # Long test string; further breaking harms readability.
        "When asked for a recommendation, I provided a data-backed conclusion favoring gradient boosting based on precisely quantified operational advantages in speed, robustness to the 8% missing data rate in production, and interpretability benefits, despite the neural network's marginal accuracy gains."  # Long test string; further breaking harms readability.
    ),
)

# Compile all examples into a list for optimization
analyzer_l1_examples = [example1, example2, example3, example4, example5, example6, example7]
