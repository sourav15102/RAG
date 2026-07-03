---
name: gh
description: Use when the user asks to interact with GitHub — e.g., "create a PR", "list issues", "clone a repo", "run workflows", "review code", "manage gists", or any task involving the `gh` CLI. Also use when the user says "gh", "github", "pull request", "issue", "actions", "repo", "clone", "fork", "release", "codespace", or "gist".
---

# GitHub CLI (`gh`) Skill

Comprehensive guide for using the `gh` command-line tool to interact with GitHub.

## Installation & Setup

### Install gh
```bash
# macOS (Homebrew)
brew install gh

# macOS (MacPorts)
sudo port install gh

# Linux (apt)
sudo apt install gh

# Linux (yum/dnf)
sudo dnf install 'gh'

# Verify installation
gh --version
```

### Authentication
```bash
# Interactive login (recommended)
gh auth login

# Login with token (for automation)
gh auth login --with-token < mytoken.txt

# Check auth status
gh auth status

# View which account is active
gh auth status --show-token

# Logout
gh auth logout
```

Auth methods: Browser-based OAuth (default), or personal access token (classic or fine-grained). Tokens need `repo`, `read:org`, `workflow`, `gist` scopes depending on use.

### Configuration
```bash
# Set default repository (so you can omit org/repo in commands)
gh repo set-default

# View current config
gh config list

# Set editor
gh config set editor "code --wait"

# Set git protocol
gh config set git_protocol ssh
```

## Pull Requests

### Creating PRs
```bash
# Create a PR from current branch (interactive)
gh pr create

# Create with title and body
gh pr create --title "My PR" --body "Description here"

# Create with a draft PR
gh pr create --draft

# Create and assign reviewers
gh pr create --reviewer @me --reviewer username

# Create with labels
gh pr create --label bug --label enhancement

# Create with milestone
gh pr create --milestone "v1.0"

# Create from a different branch
gh pr create --base main --head feature-branch

# Create with template
gh pr create --template pull_request_template.md

# Fill from commit messages
gh pr create --fill
```

### Reviewing PRs
```bash
# Review current PR
gh pr review

# Approve
gh pr review --approve

# Request changes
gh pr review --request-changes --body "Please fix..."

# Comment only
gh pr review --comment --body "Looks good overall, one nit:"

# Add line-specific comments
gh pr review --approve --body "LGTM" --comment "src/main.js:10:consider renaming"
```

### Merging PRs
```bash
# Merge (creates a merge commit)
gh pr merge

# Squash and merge
gh pr merge --squash --body "Squashed commit message"

# Rebase and merge
gh pr merge --rebase

# Auto-merge (set when checks pass)
gh pr merge --auto --squash

# Merge with custom subject/body
gh pr merge -t "Title" -b "Body"

# Cancel auto-merge
gh pr merge --auto --cancel
```

### Listing & Viewing PRs
```bash
# List PRs (open by default)
gh pr list

# List with filters
gh pr list --state merged
gh pr list --state closed
gh pr list --state all

# List PRs assigned to you
gh pr list --assignee @me

# List PRs you authored
gh pr list --author @me

# List PRs needing your review
gh pr list --search "review-requested:@me"

# List with labels
gh pr list --label bug

# List with limits
gh pr list --limit 50

# List in JSON format
gh pr list --json number,title,state,author,createdAt

# List with custom jq filter
gh pr list --json number,title,state --jq '.[] | "\(.number): \(.title)"'

# View a specific PR
gh pr view 42
gh pr view https://github.com/owner/repo/pull/42

# View PR in browser
gh pr view --web

# View PR diff
gh pr view --diff

# View PR comments
gh pr view --comments
```

### Checking out PRs
```bash
# Checkout PR locally
gh pr checkout 42

# Checkout PR as a new branch
gh pr checkout 42 --branch my-local-name
```

### Other PR operations
```bash
# Close a PR
gh pr close 42

# Reopen a closed PR
gh pr reopen 42

# Add comment to PR
gh pr comment 42 --body "Good catch!"

# Show PR diff stat
gh pr diff 42

# Get PR status (checks)
gh pr status
```

## Issues

### Creating Issues
```bash
# Create issue interactively
gh issue create

# Create with title and body
gh issue create --title "Bug: login fails" --body "Steps to reproduce..."

# Create with labels
gh issue create --label bug --label "high priority"

# Create with assignee
gh issue create --assignee @me

# Create with project
gh issue create --project "Project Board"

# Create with milestone
gh issue create --milestone "sprint-12"

# Create from file
gh issue create --title "Refactor X" --body-file description.md

# Create with template
gh issue create --template bug_report.md
```

### Listing Issues
```bash
# List issues (open by default)
gh issue list

# List by state
gh issue list --state closed
gh issue list --state all

# List assigned to you
gh issue list --assignee @me

# List authored by you
gh issue list --author @me

# List with label filter
gh issue list --label bug

# List with multiple label filter (AND)
gh issue list --label bug --label urgent

# List by milestone
gh issue list --milestone "sprint-12"

# List with search
gh issue list --search "error in auth"

# List in JSON
gh issue list --json number,title,labels,assignees

# Limit results
gh issue list --limit 100
```

### Viewing & Editing Issues
```bash
# View issue
gh issue view 42

# View in browser
gh issue view 42 --web

# View with comments
gh issue view 42 --comments

# View as JSON
gh issue view 42 --json title,body,comments,labels

# Add comment
gh issue comment 42 --body "Fixed in PR #43"

# Close issue
gh issue close 42 --reason completed
gh issue close 42 --reason "not planned"

# Reopen issue
gh issue reopen 42

# Edit issue
gh issue edit 42 --title "New title"
gh issue edit 42 --add-label bug
gh issue edit 42 --remove-label wontfix
gh issue edit 42 --add-assignee username
gh issue edit 42 --remove-assignee username
gh issue edit 42 --milestone "v2.0"
```

## Repositories

### Cloning
```bash
# Clone using default protocol (HTTPS or SSH based on config)
gh repo clone owner/repo

# Clone to specific directory
gh repo clone owner/repo ./my-fork

# Clone using SSH explicitly
gh repo clone owner/repo -- --depth 1
```

### Forking
```bash
# Fork a repo
gh repo fork

# Fork and clone immediately
gh repo fork --clone

# Fork to your personal account
gh repo fork --org my-org
```

### Creating Repos
```bash
# Create repo from current directory
gh repo create my-repo

# Create with description
gh repo create my-repo --description "A cool project"

# Create as public/private/internal
gh repo create my-repo --public
gh repo create my-repo --private
gh repo create my-repo --internal

# Create with remote origin
gh repo create my-repo --source=. --remote=origin --push
```

### Viewing Repos
```bash
# View repo info
gh repo view
gh repo view owner/repo

# View in browser
gh repo view --web

# View as JSON
gh repo view --json name,description,owner,stargazerCount,forkCount

# List your repos
gh repo list
gh repo list owner --limit 100

# List with language filter
gh repo list --language python

# List forks
gh repo list --fork

# List repos sorted by stars
gh repo list --limit 10 --json name,stargazerCount --jq '.[] | "\(.stargazerCount) \(.name)"'
```

### Managing Repos
```bash
# Rename repo
gh repo rename new-name

# Delete repo (destructive!)
gh repo delete

# Transfer repo
gh repo transfer new-owner

# Archive repo
gh repo archive

# Enable/disable features
gh repo edit --enable-issues=true
gh repo edit --enable-wiki=false
gh repo edit --enable-discussions=true

# Change default branch
gh repo edit --default-branch main

# Change description
gh repo edit --description "New description"

# Change homepage
gh repo edit --homepage "https://example.com"
```

## GitHub Actions

### Viewing Workflow Runs
```bash
# List recent runs
gh run list

# List runs for a specific workflow
gh run list --workflow ci.yml

# List runs by status
gh run list --status success
gh run list --status failure
gh run list --status cancelled
gh run list --status in_progress

# List run by branch
gh run list --branch main

# List runs in JSON
gh run list --json databaseId,displayTitle,status,conclusion,headBranch

# Limit results
gh run list --limit 20
```

### Viewing & Managing Runs
```bash
# View a specific run
gh run view 1234

# View run in browser
gh run view 1234 --web

# View failed jobs/steps
gh run view 1234 --log-failed

# View specific job logs
gh run view --job 5678

# Watch run in real-time
gh run watch 1234

# Rerun a failed workflow
gh run rerun 1234

# Rerun only failed jobs
gh run rerun 1234 --failed

# Cancel a run
gh run cancel 1234

# Download run logs
gh run download 1234
```

### Triggering Workflows
```bash
# Trigger workflow dispatch event
gh workflow run ci.yml

# Trigger with input parameters
gh workflow run deploy.yml --ref main --field environment=production --field version=v1.2

# Trigger with JSON input
gh workflow run test.yml --json-input '{"os": ["ubuntu", "macos"]}'

# List workflows
gh workflow list

# List workflow in JSON
gh workflow list --json name,path,state
```

## Releases

### Creating Releases
```bash
# Create a release
gh release create v1.0.0

# Create with title and notes
gh release create v1.0.0 --title "v1.0.0" --notes "First stable release"

# Create from git tag notes
gh release create v1.0.0 --notes-start-tag v0.9.0

# Create with files attached
gh release create v1.0.0 ./dist/app.zip ./dist/app.dmg

# Create as draft
gh release create v1.0.0 --draft

# Create as prerelease
gh release create v1.0.0-rc1 --prerelease

# Create from notes file
gh release create v1.0.0 --notes-file CHANGELOG.md

# Generate notes from conventional commits
gh release create v1.0.0 --generate-notes
```

### Viewing & Downloading Releases
```bash
# List releases
gh release list

# List with limit
gh release list --limit 5

# View release
gh release view v1.0.0

# View in browser
gh release view v1.0.0 --web

# View assets in JSON
gh release view v1.0.0 --json assets

# Download release assets
gh release download v1.0.0

# Download to specific directory
gh release download v1.0.0 --dir ./downloads

# Download specific files (glob pattern)
gh release download v1.0.0 --pattern "*.dmg"

# Download latest release
gh release download --pattern "*.zip"

# View release notes
gh release view v1.0.0 --json body
```

### Deleting Releases
```bash
# Delete a release
gh release delete v1.0.0
```

## Gists

### Creating Gists
```bash
# Create a public gist
gh gist create file.py

# Create a secret gist
gh gist create file.py --public   # actually makes it public; default is secret

# Create with description
gh gist create file.py --desc "Utility script for parsing CSV"

# Create from multiple files
gh gist create file1.py file2.py

# Create from stdin
echo "console.log('hello')" | gh gist create --filename hello.js
```

### Listing & Viewing Gists
```bash
# List your gists
gh gist list

# List with limit
gh gist list --limit 20

# List publicly (not just yours)
gh gist list --public

# View gist
gh gist view GIST_ID

# View in browser
gh gist view GIST_ID --web

# View with raw content
gh gist view GIST_ID --raw

# View as JSON
gh gist view GIST_ID --json files,description,createdAt
```

### Editing & Deleting Gists
```bash
# Edit gist (opens editor)
gh gist edit GIST_ID

# Edit with new file content
gh gist edit GIST_ID --add updated_file.py

# Delete gist
gh gist delete GIST_ID
```

## Codespaces

```bash
# List codespaces
gh codespace list

# Create a codespace
gh codespace create
gh codespace create --repo owner/repo --branch main

# Connect to a codespace
gh codespace ssh

# Open in VS Code
gh codespace code

# Open in browser
gh codespace open

# Stop codespace
gh codespace stop

# Delete codespace
gh codespace delete

# View codespace details
gh codespace view

# Rename codespace
gh codespace edit --display-name "My dev env"

# Port forwarding
gh codespace ports
gh codespace ports visibility 8080:public

# Rebuild container
gh codespace rebuild

# View logs
gh codespace logs
```

## Search

```bash
# Search repositories
gh search repos "machine learning" --limit 20
gh search repos --owner microsoft --language python
gh search repos --topic ai --topic nlp
gh search repos --stars ">=1000"
gh search repos --updated ">2024-01-01"

# Search issues
gh search issues "bug" --repo owner/repo
gh search issues --label "good first issue" --state open
gh search issues --author @me
gh search issues --assignee @me

# Search pull requests
gh search prs "refactor" --repo owner/repo
gh search prs --state merged --author @me
gh search prs --review-requested @me

# Search code
gh search code "function parseAuth" --repo owner/repo
gh search code "TODO" --language python --owner microsoft

# Search in JSON format
gh search repos "awesome" --json fullName,stargazerCount --jq '.[] | "\(.fullName): \(.stargazerCount)⭐"'
```

## Raw API Access

```bash
# GET request
gh api /repos/owner/repo

# GET with pagination
gh api /repos/owner/repo/issues --paginate

# POST request
gh api /repos/owner/repo/issues --method POST --field title="New Issue" --field body="Body text"

# POST with JSON body
gh api /repos/owner/repo/issues --method POST --input body.json

# PATCH request
gh api /repos/owner/repo/issues/42 --method PATCH --field title="Updated title"

# DELETE request
gh api /repos/owner/repo/releases/12345 --method DELETE

# Set headers
gh api /user --header "Accept: application/vnd.github+json"

# Filter with jq
gh api /repos/owner/repo/pulls --jq '.[] | {number, title, state}'

# Output raw (without newline at end)
gh api /repos/owner/repo/contents/README.md --jq '.content' --raw
```

### Common API endpoints
```bash
# Get authenticated user
gh api /user

# List repo issues
gh api /repos/owner/repo/issues

# List repo collaborators
gh api /repos/owner/repo/collaborators

# List repo branches
gh api /repos/owner/repo/branches

# List workflow runs
gh api /repos/owner/repo/actions/runs

# List commits
gh api /repos/owner/repo/commits

# List releases
gh api /repos/owner/repo/releases

# Create a label
gh api /repos/owner/repo/labels --method POST --field name=bug --field color=d73a4a

# Get repo stats
gh api /repos/owner/repo --jq '{stars: .stargazers_count, forks: .forks_count, issues: .open_issues_count}'
```

## Advanced & Useful Patterns

### Working with JSON output
```bash
# Get specific JSON fields
gh pr list --json number,title,author,createdAt

# Pipe through jq for custom formatting
gh pr list --json number,title,author --jq '.[] | "#\(.number): \(.title) by \(.author.login)"'

# Complex jq filtering
gh issue list --json labels,title --jq '.[] | select(.labels[].name == "bug") | .title'

# Output count only
gh issue list --json number --jq 'length'
```

### Automation & scripting
```bash
# Check if a PR has been merged
gh pr view 42 --json merged --jq '.merged'

# Get the default branch
gh repo view --json defaultBranch --jq '.defaultBranch'

# Wait for checks to pass
gh pr checks 42 --watch

# Check CI status before merge
gh pr merge 42 --auto --squash

# Bulk close issues by label
gh issue list --label stale --json number --jq '.[].number' | xargs -I{} gh issue close {}

# Bulk add label
gh issue list --search "bug" --json number --jq '.[].number' | xargs -I{} gh issue edit {} --add-label triaged

# Approve and merge in one step
gh pr review --approve && gh pr merge --squash

# Create PR from current branch with conventional commit format
gh pr create --title "feat: add user authentication" --body "## Summary\n\nImplements OAuth login flow." --label enhancement

# Get the latest release tag
gh release list --limit 1 --json tagName --jq '.[0].tagName'
```

### Environment variables
```bash
# Set token for automation (instead of auth login)
export GH_TOKEN=github_pat_...

# Override the default host (for GitHub Enterprise)
export GH_HOST=github.mycompany.com

# Set pager
export GH_PAGER=less

# Disable prompting for terminal
export GH_PROMPT_DISABLED=1
```

### Tips & Best Practices
- Always use `--json` + `--jq` instead of parsing human-readable output in scripts
- Use `--paginate` for endpoints that return >30 items
- Prefer SSH protocol (`gh config set git_protocol ssh`) for clone/push operations
- Use `gh repo set-default` to save typing `owner/repo` in subsequent commands
- For GitHub Enterprise, set `GH_HOST` or configure via `gh config set host`
- Use `gh workflow run` with `--ref` to trigger on a specific branch
- Tab-completion is available: `gh completion -s zsh > /usr/local/share/zsh/site-functions/_gh`
- `gh` never stores your token in plain text — it uses the OS keyring (macOS Keychain, Linux `secret-tool`, Windows Credential Manager)
