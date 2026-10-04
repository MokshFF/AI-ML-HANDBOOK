"""Cheat Sheet Parser and Reference Utility."""
from pathlib import Path
from typing import List, Dict, Any

CS_DIR = Path(__file__).resolve().parent.parent

def list_cheat_sheets() -> List[str]:
    return sorted([f.name for f in CS_DIR.glob("*.md") if f.name != "README.md" and f.name != "interview.md" and f.name != "references.md"])

def inspect_cheat_sheet(filename: str) -> Dict[str, Any]:
    file_path = CS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Cheat sheet not found: {filename}")
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    title = lines[0].replace("#", "").strip() if lines else ""
    return {
        "filename": filename,
        "title": title,
        "line_count": len(lines),
        "table_rows": len([l for l in lines if l.startswith("|")])
    }

if __name__ == "__main__":
    sheets = list_cheat_sheets()
    print(f"Discovered {len(sheets)} cheat sheets:")
    for s in sheets:
        info = inspect_cheat_sheet(s)
        print(f" - {info['filename']}: {info['title']} ({info['table_rows']} table rows)")
