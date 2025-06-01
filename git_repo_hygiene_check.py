
import subprocess
import os

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def check_untracked():
    print("\n=== Checking for untracked files ===")
    run_cmd("git status")

def check_ignored_files():
    print("\n=== Checking for files that match common junk patterns ===")
    junk_patterns = [
        "*.pyc", "*.pyo", "*.log", "*.tmp", ".DS_Store", "*.npy",
        "*.code-workspace", "*.sample", "*.TAG", "*.cfg", "*.csv", "*.pdf",
        "git_repo_hygiene_check.py"
    ]
    pattern_str = " -o ".join([f"-name \"{p}\"" for p in junk_patterns])
    find_cmd = f"find . -type f \( {pattern_str} \)"
    run_cmd(find_cmd)

def confirm_gitignore():
    print("\n=== Confirming .gitignore presence and contents ===")
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            print(f.read())
    else:
        print(".gitignore file not found!")

def list_tracked_files():
    print("\n=== Listing tracked files ===")
    run_cmd("git ls-files")

def run_git_health_checks():
    print("\n=== Running basic git health checks ===")
    run_cmd("git gc")
    run_cmd("git fsck")

if __name__ == "__main__":
    check_untracked()
    check_ignored_files()
    confirm_gitignore()
    list_tracked_files()
    run_git_health_checks()
