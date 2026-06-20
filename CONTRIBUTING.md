# Contributing Guide (Windows)

Welcome to contribute to this project!

This document will guide you through the complete development workflow, even if you are new to Git and GitHub.

---

# Project Collaboration Workflow

## Branch Types

You can see the current branch in the top-left corner of the GitHub repository page.

| Branch | Purpose |
|---|---|
| `main` | Stable production-ready version. Do NOT modify directly. |
| `develop` | Daily development branch. |
| `feature/*` | Branches for developing new features. |

Example:

```text
feature/login-page
feature/image-upload
feature/model-training
```

---

## Development Procedure

The normal workflow is:

```text
clone → create branch → modify → commit → push → Pull Request
```

---

# Initial Preparation

## 1. Install Git

Download Git from:

```text
https://git-scm.com/download/win
```

During installation:

```text
Just keep clicking "Next"
```

Git is the foundational tool for all following operations.

---

## 2. Configure Your Git Identity

Open **Git Bash** and run:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Please replace them with your own GitHub username and email.

You can verify your configuration using:

```bash
git config --global --list
```

---

## 3. Connect GitHub Account Using SSH

### Generate SSH Key

Run:

```bash
ssh-keygen -t ed25519 -C "you@example.com"
```

Please replace it with your own GitHub email.

When prompted:

```text
Press Enter for all questions
```

---

### Check Your Public Key

Run:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the entire output.

---

### Add SSH Key to GitHub

Go to:

```text
GitHub
→ Settings (Top right corner)
→ SSH and GPG keys (Left column)
→ New SSH key
```

Paste your public key and save it.

---

### Test SSH Connection

Run:

```bash
ssh -T git@github.com
```

If successful, you should see something similar to:

```text
Hi username! You've successfully authenticated...
```

---

# Clone the Project

Choose a folder on your PC, such as a `Projects` folder under your user directory.

Open Git Bash inside that folder and run:

```bash
git clone git@github.com:fakedpotato1/insect-recognition-ai.git
cd insect-recognition-ai
```

This will download the project to your local machine.

---

# Create Your Development Branch

Never develop directly on `main`.

Create your own feature branch:

```bash
git checkout -b feature/your-feature-name
```

Example:

```bash
git checkout -b feature/login-page
```

---

# Branch Naming Rules

Use clear and simple branch names.

Format:

```text
feature/short-description
fix/short-description
docs/short-description
```

Examples:

```text
feature/login-page
feature/image-upload
fix/upload-bug
docs/update-readme
```

Avoid names like:

```text
aaa
test
mybranch
new
```

Branch names should describe what you are working on.

---

# Start Development

Now you can modify the project files using your preferred editor or IDE.

---

# Save Your Changes

## Check Modified Files

```bash
git status
```

---

## Add Files

Add **all** modified files:

```bash
git add .
```

Or add a specific file:

```bash
git add filename.py
```

You can get the file path by right-clicking the file and paste it here.

---

## Commit Your Changes

```bash
git commit -m "feat: add login page"
```

### Common Commit Prefixes

| Prefix | Meaning |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `refactor` | Code refactoring |
| `style` | Formatting/style changes |
| `test` | Testing related |

Examples:

```bash
git commit -m "fix: solve image upload bug"
git commit -m "docs: update README"
```

If you don't know how to write it, ask AI. (What is **this**(you can describe your change in short for AI) conventional commit?)

---

# Push Your Branch

Push your branch to GitHub:

```bash
git push origin feature/your-feature-name
```

**feature/your-feature-name** is the branch you are working.

Example:

```bash
git push origin feature/login-page
```

---

# Create a Pull Request (PR)

After pushing:

1. Open the GitHub repository page
2. GitHub will show:
   ```text
   Compare & pull request
   ```
3. Click it
4. Describe your changes
5. Submit the Pull Request

Your code will then be reviewed before merging.

---

# Update Your Local Repository

Before starting new work, **always** pull the latest changes:

```bash
git checkout develop
git pull origin develop
```

Then return to the branch where you work.

```bash
git checkout your-branch
```

Please replace **your-branch** by your working branch.

---

# Common Problems

## Permission denied (publickey)

Reason:

```text
SSH key is not configured correctly
```

Solution:

- Recheck SSH key setup
- Ensure the key is added to GitHub

---

## rejected non-fast-forward

Reason:

```text
Remote branch contains newer commits
```

Solution:

```bash
git pull origin develop
```

Then try pushing again.

---

## fatal: not a git repository

Reason:

```text
You are not inside the project folder
```

Solution:

```bash
cd insect-recognition-ai
```

---

# Important Rules

- Never push directly to `main`
- Always create your own feature branch
- Write meaningful commit messages
- Pull latest changes before starting new work
- Make small and clear commits whenever possible

---

Thank you for contributing!
