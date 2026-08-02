#!/usr/bin/env python3
"""Delete real_* and fake_* directories inside a tmp folder.

Usage examples:
  python clean_tmp.py
  python clean_tmp.py /path/to/tmp
  python clean_tmp.py /path/to/tmp --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def remove_matching_dirs(tmp_dir: Path, dry_run: bool = False) -> tuple[list[Path], list[Path]]:
	"""Remove directories matching real_* and fake_* under tmp_dir.

	Returns:
		Tuple of (deleted_or_would_delete, skipped_paths).
	"""
	deleted: list[Path] = []
	skipped: list[Path] = []

	for pattern in ("real_*", "fake_*"):
		for path in tmp_dir.glob(pattern):
			if not path.is_dir():
				skipped.append(path)
				continue

			if dry_run:
				deleted.append(path)
				continue

			shutil.rmtree(path)
			deleted.append(path)

	return deleted, skipped


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Delete real_* and fake_* directories inside a tmp folder."
	)
	parser.add_argument(
		"tmp_dir",
		nargs="?",
		default="/tmp",
		help="Path to tmp folder (default: ./tmp)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show directories that would be deleted without deleting them.",
	)
	args = parser.parse_args()

	tmp_dir = Path(args.tmp_dir).expanduser().resolve()
	if not tmp_dir.exists():
		print(f"tmp folder does not exist: {tmp_dir}")
		return
	if not tmp_dir.is_dir():
		print(f"path is not a directory: {tmp_dir}")
		return

	deleted, skipped = remove_matching_dirs(tmp_dir, dry_run=args.dry_run)

	if args.dry_run:
		print("Dry run mode: no directories were deleted.")

	if deleted:
		action = "Would delete" if args.dry_run else "Deleted"
		for p in deleted:
			print(f"{action}: {p}")
	else:
		print("No matching directories found.")

	if skipped:
		for p in skipped:
			print(f"Skipped (not a directory): {p}")


if __name__ == "__main__":
	main()
