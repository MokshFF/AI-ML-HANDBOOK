"""Tests for Cheat Sheet Integrity."""
from cheat_sheet_viewer import list_cheat_sheets, inspect_cheat_sheet

EXPECTED_SHEETS = [
    "01-ml-algorithms.md",
    "02-ml-metrics.md",
    "03-dl-architectures.md",
    "04-optimization.md",
    "05-nlp.md",
    "06-transformers.md",
    "07-rag.md",
    "08-llms.md",
    "09-mlops.md",
    "10-system-design.md",
]

def test_all_cheat_sheets_exist():
    available = list_cheat_sheets()
    for exp in EXPECTED_SHEETS:
        assert exp in available, f"Missing cheat sheet: {exp}"

def test_cheat_sheet_content():
    for exp in EXPECTED_SHEETS:
        info = inspect_cheat_sheet(exp)
        assert info["line_count"] >= 5
        assert info["table_rows"] > 3
