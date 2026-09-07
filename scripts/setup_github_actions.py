"""
GitHub Actions Deployment Setup

Interactive script that helps you:
1. Initialize a git repo (if needed)
2. Create the GitHub Actions workflow
3. Push to GitHub
4. Verify the setup

Usage:
  python scripts/setup_github_actions.py
"""

import subprocess
from pathlib import Path


def run(cmd: str, check: bool = True) -> tuple[int, str]:
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=60
    )
    if check and result.returncode != 0:
        print(f"  [FAIL] Command failed: {cmd}")
        print(f"  Error: {result.stderr}")
    return result.returncode, result.stdout.strip()


def check_git() -> bool:
    """Check if git is installed and repo exists."""
    code, _ = run("git --version", check=False)
    if code != 0:
        print("[FAIL] Git is not installed. Install it from https://git-scm.com")
        return False

    code, _ = run("git status", check=False)
    if code != 0:
        print("[DIR] Initializing git repository...")
        run("git init")
        run("git checkout -b main")

    return True


def create_workflow():
    """Create the GitHub Actions workflow file."""
    workflow_dir = Path(".github/workflows")
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_file = workflow_dir / "bot.yml"

    # Read the template
    template = Path("deploy/github_actions.yml")
    if template.exists():
        content = template.read_text()
    else:
        content = """name: Trading Bot Runner
on:
  schedule:
    - cron: '0 */4 * * *'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run-bot:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Run bot
        env:
          BOT_MODE: paper
          EXCHANGE_API_KEY: ${{ secrets.EXCHANGE_API_KEY }}
          EXCHANGE_API_SECRET: ${{ secrets.EXCHANGE_API_SECRET }}
        # run_github_bot.py is deprecated; the scheduled runner is paper_trader.py
        run: python -u scripts/paper_trader.py
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: trade-log-${{ github.run_number }}
          path: data/results/github_bot_log.jsonl
          retention-days: 90
"""

    workflow_file.write_text(content)
    print(f"  [OK] Created {workflow_file}")


def setup_repo():
    """Help user set up the GitHub repo."""
    print()
    print("Step 1: Create a GitHub Repository")
    print("-" * 40)
    print("  1. Go to https://github.com/new")
    print("  2. Name it (e.g., 'trading-bot')")
    print("  3. Keep it Private (recommended)")
    print("  4. Don't initialize with README")
    print()

    repo_url = input("Paste the repo URL (e.g., https://github.com/user/trading-bot.git): ").strip()

    if repo_url:
        # Add remote
        code, _ = run("git remote -v", check=False)
        if "origin" in code:
            print("  [WARN]️  Remote 'origin' already exists. Updating...")
            run(f"git remote set-url origin {repo_url}")
        else:
            run(f"git remote add origin {repo_url}")

        print(f"  [OK] Remote added: {repo_url}")
        return True

    return False


def commit_and_push():
    """Commit all files and push."""
    print()
    print("Step 3: Commit and Push")
    print("-" * 40)

    # Check if there's anything to commit
    code, status = run("git status --porcelain", check=False)
    if not status:
        print("  [INFO]️  No changes to commit.")
        # Try to push anyway
        code, _ = run("git push -u origin main", check=False)
        if code == 0:
            print("  [OK] Pushed to GitHub!")
        else:
            print("  [FAIL] Push failed. Try: git push -u origin main")
        return

    run("git add .")
    run('git commit -m "Initial trading bot deployment"')
    run("git branch -M main")

    print("  [*] Pushing to GitHub...")
    code, _ = run("git push -u origin main", check=False)
    if code == 0:
        print("  [OK] Pushed successfully!")
    else:
        print("  [FAIL] Push failed.")
        print("  Try manually: git push -u origin main")


def print_next_steps():
    """Print the remaining manual steps."""
    print()
    print("=" * 60)
    print("  NEXT STEPS (Manual)")
    print("=" * 60)
    print()
    print("Step 2: Add API Keys as Secrets")
    print("-" * 40)
    print("  1. Go to your repo → Settings → Secrets and variables → Actions")
    print("  2. Click 'New repository secret'")
    print("  3. Add these secrets:")
    print()
    print("     Name: EXCHANGE_API_KEY")
    print("     Value: your Binance API key")
    print()
    print("     Name: EXCHANGE_API_SECRET")
    print("     Value: your Binance API secret")
    print()
    print("Step 4: Enable the Workflow")
    print("-" * 40)
    print("  1. Go to Actions tab in your repo")
    print("  2. Click 'I understand my workflows, go ahead and enable them'")
    print("  3. The bot will run automatically every 4 hours!")
    print()
    print("Step 5: Monitor")
    print("-" * 40)
    print("  • Check runs: Actions tab → click any workflow")
    print("  • Download logs: Actions → run → Artifacts → trade-log-*")
    print("  • Manual trigger: Actions → Run workflow")
    print()
    print("To go live later:")
    print("  • Edit .github/workflows/bot.yml: BOT_MODE: live")
    print("  • Commit and push: git add . && git commit -m 'go live' && git push")
    print()


def main():
    print("=" * 60)
    print("  GITHUB ACTIONS DEPLOYMENT SETUP")
    print("=" * 60)
    print()
    print("This script will help you deploy your trading bot to GitHub Actions.")
    print("The bot will run every 4 hours for FREE (no credit card needed).")
    print()

    # Check prerequisites
    if not check_git():
        return

    # Create workflow
    print()
    print("Step 1: Creating GitHub Actions workflow...")
    create_workflow()

    # Setup repo
    print()
    print("Step 2: Connect to GitHub")
    print("-" * 40)

    answer = input("Do you have a GitHub repo ready? (y/n): ").strip().lower()
    if answer == "y":
        setup_repo()
        commit_and_push()
    else:
        print()
        print("  Create one now at: https://github.com/new")
        print("  Then re-run this script.")
        print()
        return

    print_next_steps()


if __name__ == "__main__":
    main()
