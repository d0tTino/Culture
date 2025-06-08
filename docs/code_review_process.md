# Culture.ai Code Review Process

## 1. Purpose and Goals

Code reviews are a fundamental practice that helps us achieve several important goals:

- **Improve Code Quality**: Identify bugs, logic errors, and edge cases before they reach production
- **Ensure Adherence to Standards**: Maintain consistent coding style and patterns across the codebase
- **Share Knowledge**: Spread understanding of the codebase across the team and prevent knowledge silos
- **Facilitate Learning**: Both authors and reviewers learn from the review process
- **Promote Best Practices**: Ensure proper architecture, security, and performance considerations

## 2. Scope of Reviews

### What Should Be Reviewed

- All new features and functionality
- Bug fixes
- Significant refactoring of existing code
- Changes to core components or critical paths
- Changes to the public API or interfaces

### What May Skip Formal Review

- Very minor typo fixes in comments or documentation
- Simple formatting changes (though these should typically be handled by automated tools)
- Documentation-only changes that don't affect code behavior or API descriptions
- Experimental code in isolated branches not intended for merging to main

## 3. Reviewer Assignment

Given our current team structure:

- All significant code changes made by Claude will be conceptually "reviewed" by the Dev Lead (via the Operator)
- For critical components or complex changes, consider scheduling a synchronous review session
- As the team grows, aim to have at least one reviewer familiar with the component being modified

## 4. Review Process

### Author's Responsibilities (Pre-Submission)

Before submitting code for review, the author should:

1. **Self-Review**: Perform a self-review of the code using the same criteria a reviewer would
2. **Follow Standards**: Ensure code adheres to `docs/coding_standards.md`
3. **Run Linters**: Execute `scripts/lint.sh --format` (or `scripts/lint.bat --format`) and address all reported issues
4. **Test Coverage**: Write and execute tests for the changes, ensuring all tests pass
5. **Clear Description**: Provide a clear description of the changes, including:
   - What problem the change solves
   - How the solution works at a high level
   - Any trade-offs made and why
   - Any areas where the author is uncertain or would like specific feedback
6. **Keep Changes Focused**: Submit smaller, focused changes rather than large, sweeping changes

### Reviewer's Responsibilities (During Review)

When reviewing code, focus on these key areas:

#### Correctness
- Does the code correctly implement the intended functionality?
- Are edge cases handled appropriately?
- Does the code address the requirements completely?
- Is the logic sound and free of bugs?

#### Readability & Maintainability
- Is the code clear and easy to understand?
- Are variables, functions, and classes named meaningfully?
- Are complex sections adequately commented?
- Is the code structured in a way that will be maintainable?

#### Adherence to Standards
- Does the code follow the project's coding standards as defined in `docs/coding_standards.md`?
- Are imports organized according to standards?
- Are docstrings present and properly formatted?
- Is the code properly type-hinted?

#### Test Coverage
- Are there tests for the new functionality?
- Do the tests verify both normal operation and edge cases?
- Are the tests themselves well-written and maintainable?

#### Basic Security
- Are there any obvious security issues?
- Is user input properly validated and sanitized?
- Are sensitive data (API keys, etc.) handled appropriately?

#### Basic Performance
- Are there any obvious performance issues?
- Are expensive operations optimized appropriately?
- Could any operations lead to excessive memory usage?

#### Design
- Does the change fit well with the existing architecture?
- Does it follow the project's design principles?
- Are responsibilities properly separated?
- Are appropriate patterns used?

#### Tone of Feedback
- Focus on the code, not the author
- Be specific and constructive in your feedback
- Explain the "why" behind suggestions
- Differentiate between "must fix" issues and suggestions/preferences
- Acknowledge good solutions and elegant code

#### Providing Feedback
- Use clear, specific comments
- Suggest alternatives when identifying issues
- Ask questions to understand the author's intent when something isn't clear
- Provide references to relevant documentation or examples when applicable

### Author's Responsibilities (Post-Review)

After receiving review feedback:

1. **Address All Feedback**: Respond to all comments, either by making changes or explaining why changes aren't appropriate
2. **Discuss Constructively**: If disagreeing with feedback, explain your reasoning clearly and be open to discussion
3. **Update the Code**: Make necessary changes based on the review
4. **Request Re-review**: Indicate when changes are ready for another review, highlighting significant changes made
5. **Learn and Improve**: Use feedback to improve future code submissions

## 5. Branching Strategy

To support the review process, we recommend the following branching strategy:

- The `main` branch is protected and represents production-ready code
- Create new branches for features, bugfixes, or other work:
  - Feature branches: `feature/descriptive-feature-name`
  - Bug fix branches: `fix/brief-bug-description`
  - Refactoring branches: `refactor/component-being-refactored`
- Work on these branches and submit for review when ready
- After successful review, changes are merged into `main`
- Delete branches once they've been merged

## 6. Common Review Checklist

### General
- [ ] Code follows the project's coding standards
- [ ] Functions and methods do one thing and do it well
- [ ] No code duplication or unnecessary complexity
- [ ] No commented-out code (unless explicitly explained)
- [ ] Error handling is appropriate and consistent

### Python Specific
- [ ] Type hints used appropriately
- [ ] Docstrings present and formatted according to standards
- [ ] Appropriate use of Python idioms
- [ ] Follows PEP 8 guidelines as defined in our standards
- [ ] Uses appropriate data structures for the task

### Testing
- [ ] Tests exist for new functionality
- [ ] Tests verify both normal cases and edge cases
- [ ] Tests are clear and maintainable

### Documentation
- [ ] User-facing changes are documented
- [ ] Complex algorithms or decisions have explanatory comments
- [ ] Public APIs are documented

## 7. Tools and Resources

 - Linting: `scripts/lint.sh --format` or `scripts/lint.bat --format`
- Code Standards: `docs/coding_standards.md`
- Python Style Guide: [PEP 8](https://peps.python.org/pep-0008/)
- Type Hinting Guide: [PEP 484](https://peps.python.org/pep-0484/)

## 8. Continuous Improvement

This code review process is intended to evolve over time. As the project grows, we will:

- Periodically review and refine the review process
- Consider implementing automated code review tools
- Adjust the scope and depth of reviews based on project needs

Suggestions for improving this process are always welcome. 
