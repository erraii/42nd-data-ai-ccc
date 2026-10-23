#!/usr/bin/env python3
"""Select the starting position used by the environment and training scripts."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", choices=["level3", "level4"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "scenes" / f"{args.level}.json"
    destination = root / "colony_one/scene.json"
    shutil.copyfile(source, destination)
    print(f"Selected {args.level}: {destination}")


if __name__ == "__main__":
    main()
