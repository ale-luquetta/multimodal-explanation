"""Minimal interactive numbered-menu helpers."""

from pathlib import Path


def pick_option(title: str, options: list[str], back_label: str = "Back") -> int | None:
    """Render a numbered menu and return the chosen index (0-based).

    Returns None if the user picks the final "Back" entry.
    """
    print(f"\n=== {title} ===")
    for i, label in enumerate(options, start=1):
        print(f"{i}) {label}")
    print(f"{len(options) + 1}) {back_label}")

    while True:
        raw = input("Choice: ").strip()
        if not raw.isdigit():
            print("Please enter a number.")
            continue
        choice = int(raw)
        if choice == len(options) + 1:
            return None
        if 1 <= choice <= len(options):
            return choice - 1
        print("Out of range. Try again.")


def pick_yaml_from(directory: Path, title: str) -> Path | None:
    """List *.yaml files under `directory` and let the user pick one."""
    yamls = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not yamls:
        print(f"\nNo YAML configs found in {directory}. Add one and try again.")
        return None
    idx = pick_option(title, [p.name for p in yamls])
    return None if idx is None else yamls[idx]
