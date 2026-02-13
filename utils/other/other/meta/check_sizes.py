#!/usr/bin/env python3
import os

# ---- LOGIC ----
def human(n: int) -> str:
    """Return a human-readable size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.2f} TB"

def count_n_files_in_dir(path: str) -> int:
    """Return number of files in a directory (including its subfolders)."""
    total_files = 0
    for dirpath, _, filenames in os.walk(path, followlinks=False):
        total_files += len(filenames)
    return total_files

def get_dir_size(path: str) -> int:
    """Return total size of a directory (including its files and subfolders)."""
    total = 0
    for dirpath, _, filenames in os.walk(path, followlinks=False):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except (FileNotFoundError, PermissionError):
                    pass
    return total

def list_subdirs(base_dir: str):
    """Return first- and second-level subdirectories with their sizes."""
    result = {}
    for root, dirs, _ in os.walk(base_dir):
        depth = root[len(base_dir):].count(os.sep)
        if depth >= n_depth: # NOTE: set depth limit here
            dirs[:] = []
            continue
        for d in dirs:
            full_path = os.path.join(root, d)
            size = get_dir_size(full_path)
            result[full_path] = size
    return result

def print_hierarchy(base_dir: str):
    """Print the first- and second-level directories with nice tree-style indentation."""
    subdirs = list_subdirs(base_dir)

    # Print total amount of bytes in the base directory
    total_size = get_dir_size(base_dir)
    print("\n" + "=" * 80)
    print(f"Total size of '{base_dir}': {human(total_size)} ({total_size:,} bytes)")
    print("=" * 80)

    # organize into hierarchy: {parent: [(child, size), ...]}
    tree = {}
    for path, size in subdirs.items():
        parent = os.path.dirname(path)
        tree.setdefault(parent, []).append((path, size))

    def print_level(parent, prefix=""):
        if parent not in tree:
            return
        # sort children by size descending
        children = sorted(tree[parent], key=lambda x: x[1], reverse=True)
        for i, (path, size) in enumerate(children):
            name = os.path.basename(path)
            connector = "└── " if i == len(children) - 1 else "├── "
            print(f"{prefix}{connector}{name:<30} {human(size):>10}  ({size} bytes) -- {count_n_files_in_dir(path)} files")
            if path in tree:  # has subfolders
                new_prefix = prefix + ("    " if i == len(children) - 1 else "│   ")
                print_level(path, new_prefix)

    print(f"\nDirectory hierarchy for: {base_dir}\n")
    print_level(base_dir)

    # Print 10 largest directories overall
    print("\n" + "=" * 80)
    print("Top 10 Largest Directories (Overall)".center(80))
    print("=" * 80)
    print(f"{'Rank':<5} {'Directory Path':<55} {'Size':>10}  {'(Bytes)':>12}")
    print("-" * 80)

    largest = sorted(subdirs.items(), key=lambda x: x[1], reverse=True)[:10]

    for i, (path, size) in enumerate(largest, start=1):
        short_path = (
            path if len(path) <= 55 else "..." + path[-52:]
        )  # shorten long paths for display
        print(f"{i:<5} {short_path:<55} {human(size):>10}  ({size:>12,d})")

    print("=" * 80 + "\n")

# ---- MAIN ----
def main():
    if not os.path.isdir(BASE_DIR):
        print(f"Not a directory: {BASE_DIR}")
        return
    print_hierarchy(BASE_DIR)

if __name__ == "__main__":

    # ---- CONFIG ----
    BASE_DIR = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF"
    n_depth = 5

    # Run main
    main()
