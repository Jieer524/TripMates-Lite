"""Move the two tracked legacy Overpass snapshots into their archive path."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cache"
TARGET = ROOT / "archive" / "overpass_cache"
FILES = (
    "07289a46822b0a9de1ec4742655e7e15fd36fa70.json",
    "f5395737bcb7e1f88e883e52eb64bac68dee72ea.json",
)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        source = SOURCE / name
        target = TARGET / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(target)
        shutil.move(str(source), str(target))
    if SOURCE.exists() and not any(SOURCE.iterdir()):
        SOURCE.rmdir()
    print(f"Archived {len(FILES)} legacy Overpass snapshots to {TARGET}")


if __name__ == "__main__":
    main()
