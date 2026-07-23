# -*- coding: utf-8 -*-
"""
JSON Structure Explorer

Reads a JSON file and prints/writes the complete structure.

@author: Adrian
"""

import json
from pathlib import Path

# =====================================================
# INPUT JSON
# =====================================================

JSON_FILE = Path(
    r"C:\Users\DiazdeRiveraA\OneDrive - DB E.C.O. North America\2-4025 VIA Rail Expert Support - Documents\05 WORK PACKAGES\12 EXO RTC Model\04 - CN Traffic\TOR_MNT_base_Plant-input-UC-3.json"
)

OUTPUT_FILE = JSON_FILE.with_suffix(".structure.txt")


# =====================================================
# RECURSIVE SCANNER
# =====================================================

def scan_json(obj, level=0, name="ROOT", output=None):
    """
    Recursively scan JSON structure.
    """

    indent = "    " * level

    # Dictionary
    if isinstance(obj, dict):

        output.append(
            f"{indent}{name} [dict] ({len(obj)} keys)"
        )

        for key, value in obj.items():
            scan_json(
                value,
                level + 1,
                key,
                output
            )

    # List
    elif isinstance(obj, list):

        output.append(
            f"{indent}{name} [list] ({len(obj)} items)"
        )

        # Show first item structure only
        if len(obj) > 0:

            output.append(
                f"{indent}    Sample item structure:"
            )

            scan_json(
                obj[0],
                level + 2,
                "[0]",
                output
            )

    # Primitive values
    else:

        value_type = type(obj).__name__

        value_preview = str(obj)

        if len(value_preview) > 100:
            value_preview = value_preview[:100] + "..."

        output.append(
            f"{indent}{name} [{value_type}] = {value_preview}"
        )


# =====================================================
# MAIN
# =====================================================

def main():

    print("Loading JSON...")

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    structure = []

    scan_json(
        data,
        name="ROOT",
        output=structure
    )

    print("\n".join(structure))

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(structure))

    print("\nDone.")
    print(f"Structure saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()