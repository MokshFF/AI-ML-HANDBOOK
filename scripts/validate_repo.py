"""
Repository Structure and Integrity Validator.
Ensures every module conforms to architectural requirements.
"""

import json
import os
import re
import sys
from pathlib import Path

REQUIRED_ROOT_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "ROADMAP.md",
    "requirements.txt",
    ".gitignore",
]

REQUIRED_TOPIC_FILES = [
    "README.md",
    "notebook.ipynb",
    "code",
    "interview.md",
    "references.md",
]

MAJOR_MODULES = [
    "00-prerequisites",
    "01-machine-learning",
    "02-deep-learning",
    "03-nlp",
    "04-computer-vision",
    "05-speech-audio",
    "06-generative-ai",
    "07-mlops",
    "08-system-design",
    "09-projects",
    "10-interview-prep",
    "11-research-papers",
    "12-resources",
]

DISALLOWED_FILE_PATTERNS = [
    r"^\.env$",
    r"^\.env\..+$",
    r".*\.pem$",
    r".*\.key$",
    r"^credentials\.json$",
]

def validate_repository(root_dir: Path) -> bool:
    print(f"[*] Validating repository root at: {root_dir}")
    errors = []

    # 1. Check Root Files
    for rf in REQUIRED_ROOT_FILES:
        target = root_dir / rf
        if not target.exists():
            errors.append(f"Missing required root file: {rf}")
        else:
            print(f"  [+] Found root file: {rf}")

    # 2. Check Major Modules
    for mod in MAJOR_MODULES:
        mod_dir = root_dir / mod
        if not mod_dir.is_dir():
            errors.append(f"Missing major module directory: {mod}")
        else:
            readme = mod_dir / "README.md"
            if not readme.exists():
                errors.append(f"Missing README.md in major module: {mod}")

    # 3. Check Template Directory
    tpl_dir = root_dir / "_templates" / "topic-template"
    if not tpl_dir.is_dir():
        errors.append("Missing template directory: _templates/topic-template")
    else:
        for tf in REQUIRED_TOPIC_FILES:
            if not (tpl_dir / tf).exists():
                errors.append(f"Missing file in _templates/topic-template: {tf}")

    # 4. Check Topic Directories
    topic_count = 0
    notebook_count = 0
    for mod in MAJOR_MODULES:
        mod_dir = root_dir / mod
        if not mod_dir.is_dir():
            continue
        for sub in mod_dir.iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                topic_count += 1
                for tf in REQUIRED_TOPIC_FILES:
                    target = sub / tf
                    if not target.exists():
                        errors.append(f"Module '{mod}/{sub.name}' is missing '{tf}'")

                # Verify notebook JSON validity
                nb_path = sub / "notebook.ipynb"
                if nb_path.exists():
                    notebook_count += 1
                    try:
                        with open(nb_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if "cells" not in data or "metadata" not in data:
                            errors.append(f"Invalid notebook structure in: {nb_path}")
                        if data.get("nbformat", 0) < 4:
                            errors.append(f"Notebook nbformat < 4 in: {nb_path}")
                    except Exception as e:
                        errors.append(f"Failed to parse notebook JSON at {nb_path}: {e}")

    print(f"  [+] Validated {topic_count} topics and {notebook_count} Jupyter notebooks.")

    # 5. Check Internal Markdown Links
    print("[*] Validating internal markdown links...")
    broken_links = 0
    for md in root_dir.rglob("*.md"):
        if ".git" in md.parts or ".venv" in md.parts:
            continue
        text = md.read_text(encoding="utf-8")
        links = re.findall(r"\[([^\]]+)\]\(([^\)]+)\)", text)
        for _, link in links:
            if link.startswith("http") or link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = link.split("#")[0]
            if not clean_link:
                continue
            target = (md.parent / clean_link).resolve()
            if not target.exists():
                errors.append(f"Broken relative link in {md.relative_to(root_dir)}: {link}")
                broken_links += 1

    # 6. Check for disallowed files
    for root, _, files in os.walk(root_dir):
        rel_root = Path(root).relative_to(root_dir)
        if ".git" in rel_root.parts:
            continue
        for f in files:
            for pat in DISALLOWED_FILE_PATTERNS:
                if re.match(pat, f):
                    errors.append(f"Disallowed sensitive file tracked: {rel_root / f}")

    if errors:
        print(f"\n[-] Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return False

    print("\n[+] Validation PASSED! Repository architecture and integrity verified.")
    return True

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    success = validate_repository(repo_root)
    sys.exit(0 if success else 1)
