# Checking CI Status

Culture.ai uses GitHub Actions to run tests and linters on each branch. This repository does not include a remote by default, so you must add your GitHub remote to view CI results.

Long-running jobs execute on a self-hosted Linux runner. Each workflow also cancels previous runs on the same branch to save time.

## Adding the Remote

Use the `git remote add` command to connect your local clone to the GitHub repository:

```bash
git remote add origin https://github.com/<your-username>/Culture.git
```

Replace `<your-username>` with the appropriate account or organization.

## Viewing Status in the GitHub Interface

Once a remote is configured and pushed, navigate to the branch on GitHub. The latest workflow status appears near the top of the pull request or commit page.

## Using the `gh` CLI

Alternatively, the [GitHub CLI](https://cli.github.com/) allows querying status from the terminal:

```bash
# Show the most recent workflow run for the current branch
gh run list --limit 1
```

This displays whether the CI workflow succeeded or failed.
