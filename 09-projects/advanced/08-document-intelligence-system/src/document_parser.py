"""Document Token & Spatial Coordinate Representation."""
from typing import List, Dict, Any

SAMPLE_INVOICE_TOKENS = [
    {"text": "ACME", "box": [50, 50, 150, 80], "label": "VENDOR"},
    {"text": "SUPPLIES", "box": [160, 50, 280, 80], "label": "VENDOR"},
    {"text": "Invoice", "box": [50, 120, 120, 140], "label": "O"},
    {"text": "#", "box": [125, 120, 140, 140], "label": "O"},
    {"text": "INV-2024-91", "box": [145, 120, 250, 140], "label": "INVOICE_ID"},
    {"text": "Date:", "box": [50, 160, 100, 180], "label": "O"},
    {"text": "2024-10-15", "box": [105, 160, 210, 180], "label": "DATE"},
    {"text": "Total:", "box": [50, 400, 110, 425], "label": "O"},
    {"text": "$1,450.00", "box": [120, 400, 220, 425], "label": "TOTAL"}
]

ENTITY_MAP = {"O": 0, "VENDOR": 1, "INVOICE_ID": 2, "DATE": 3, "TOTAL": 4}
REV_ENTITY_MAP = {v: k for k, v in ENTITY_MAP.items()}
