# ===============================================
# Extended Metadata Analyzer for IIS_BIOBIZKAIA
# Computes:
#   - #files per type
#   - Non-empty lines per type (+ % non-empty)
#   - Total size in MB
#   - Visual progress bar during scanning
# Additionally:
#   - Finds files changed between last Monday and today (filesystem and optional git methods)
# Some characteristics:
#  - Counts ONLY non-empty lines; optional comment filtering per language
#  - Handles .owl (RDF/XML) correctly, including <!-- --> blocks
#  - Detects & skips likely-classification files for line stats
#  - Includes files with NO extension
#  - Excludes paths that start with exclude_dir
# ===============================================

from pathlib import Path
from collections import defaultdict, OrderedDict
from tqdm import tqdm  # pip install tqdm
import json
import datetime
import subprocess
import shlex
import os
from typing import Dict, List, Tuple

# --- Configuration ---
input_dir = Path("/Users/stijnvanseveren/PythonProjects/MASTERPROEF")
exclude_dir = Path("/test_FC/data/BIDS/deepprep_test")

# --- Helpers ---
def is_excluded(path: Path) -> bool:
    """Return True if 'path' should be excluded from scanning (prefix-match)."""
    try:
        return str(path).startswith(str(exclude_dir))
    except Exception:
        return False

def likely_binary(file_path: Path, sniff_bytes: int = 2048) -> bool:
    """Simple heuristic: null byte in the header => treat as classification."""
    try:
        with file_path.open("rb") as f:
            head = f.read(sniff_bytes)
        return b"\x00" in head
    except Exception:
        return True  # unreadable => treat as classification for safety

HASH_COMMENT_EXTS = {".py", ".sh", ".bash", ".zsh", ".yml", ".yaml", ".env", ".ini", ".cfg", ".conf", ".toml", ".ttl"}
SLASH_COMMENT_EXTS = {".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp", ".css", ".scss", ".go", ".rs", ".php", ".swift", ".kt"}
XML_LIKE_EXTS     = {".xml", ".html", ".xhtml", ".svg", ".xsd", ".owl"}  # OWL RDF/XML

TEXT_LIKE_EXTS = (
    HASH_COMMENT_EXTS
    | SLASH_COMMENT_EXTS
    | XML_LIKE_EXTS
    | {".json", ".txt", ".csv", ".md", ".ipynb", ".rst"}
)

def strip_line_comments(line: str, ext: str, xml_in_block: list) -> tuple[str, bool]:
    """
    Remove whole-line comments for a subset of languages.
    For XML-like, track <!-- --> block comments with a tiny state machine.
    xml_in_block is a single-item list [bool] to keep mutable state.
    Returns (possibly trimmed line, is_comment_only_line)
    """
    s = line.rstrip("\n")
    t = s.strip()

    # XML-like block comments <!-- ... -->
    if ext in XML_LIKE_EXTS:
        if xml_in_block[0]:
            # we are inside a block
            if "-->" in t:
                # close and drop the part up to -->
                xml_in_block[0] = False
                # keep whatever is after --> on the same line
                after = t.split("-->", 1)[1].strip()
                return (after, len(after) == 0)
            else:
                return ("", True)  # full line is commented
        else:
            # not inside a block
            if t.startswith("<!--"):
                if "-->" in t:
                    # single-line block comment
                    after = t.split("-->", 1)[1].strip()
                    return (after, len(after) == 0)
                else:
                    xml_in_block[0] = True
                    return ("", True)
            # not a comment line
            return (t, False)

    # Hash comments
    if ext in HASH_COMMENT_EXTS:
        if t.startswith("#"):
            return ("", True)
        return (t, False)

    # Slash comments (line //; we ignore /* */ blocks to keep it lightweight)
    if ext in SLASH_COMMENT_EXTS:
        if t.startswith("//"):
            return ("", True)
        return (t, False)

    # For other_failed text-like formats (json/csv/md/txt/ipynb...), no comment syntax
    return (t, False)

def count_nonempty_lines_in_text(file_path: Path, ext: str) -> tuple[int, int]:
    """
    Returns (non_empty, total) for text-like files.
    - For ipynb: counts non-empty lines inside code cells.
    - For others: non-empty lines, with basic whole-line comment filtering per ext.
    """
    # Special handling for notebooks: count only code cell lines
    if ext == ".ipynb":
        try:
            obj = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            nonempty = 0
            total = 0
            for cell in obj.get("cells", []):
                if cell.get("cell_type") == "code":
                    for l in cell.get("source", []):
                        total += 1
                        if l.strip():  # count non-empty only
                            nonempty += 1
            return nonempty, total
        except Exception:
            return 0, 0

    # Generic text handling
    try:
        nonempty = 0
        total = 0
        xml_in_block = [False]  # mutable flag for XML block comments
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                total += 1
                line, is_comment = strip_line_comments(raw, ext, xml_in_block)
                if line.strip() and not is_comment:
                    nonempty += 1
        return nonempty, total
    except Exception:
        return 0, 0

# --- Collect all files first (include files without extension) ---
all_files = [
    f for f in input_dir.rglob("*")
    if f.is_file() and not is_excluded(f)
]

# --- Counters ---
file_counts = defaultdict(int)
line_counts = defaultdict(int)   # non-empty
total_lines = defaultdict(int)   # total lines seen (text-like only)
total_size_mb = 0.0

# --- Main computation with progress bar ---
print(f"\n🔍 Scanning {len(all_files)} files in {input_dir} ...\n")

for file_path in tqdm(all_files, desc="Processing files", unit="file", ncols=90, dynamic_ncols=True):
    ext = file_path.suffix.lower()  # '' if no extension
    ext_label = ext if ext else "[noext]"

    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = 0.0
    total_size_mb += size_mb

    file_counts[ext_label] += 1

    # Line stats only for text-like files and non-classification
    if (ext in TEXT_LIKE_EXTS or ext == "") and not likely_binary(file_path):
        nonempty, total = count_nonempty_lines_in_text(file_path, ext)
        line_counts[ext_label] += nonempty
        total_lines[ext_label] += total

# --- Output ---
print("\n===== IIS_BIOBIZKAIA Project Metadata =====")
print(f"Total project size: {total_size_mb:.2f} MB\n")

if not file_counts:
    print("No files found.")
else:
    header = f"{'File Type':<10} {'#Files':>8} {'Non-empty Lines':>18} {'% Non-empty':>14}"
    print(header)
    print("-" * len(header))
    for ext_label, count in sorted(file_counts.items(), key=lambda x: (-x[1], x[0])):
        nonempty = line_counts.get(ext_label, 0)
        total = total_lines.get(ext_label, 0)
        pct = (nonempty / total * 100) if total else 0
        pct_str = f"{pct:>12.1f}%" if total else f"{'N/A':>14}"
        print(f"{ext_label:<10} {count:>8} {nonempty:>18} {pct_str}")

print("-" * 60)
print("Detected file data:", ", ".join(sorted(file_counts.keys())))
print("Note: [noext] = files without an extension (e.g., LICENSE, Makefile, or custom scripts).")
print("===========================================\n")

# -------------------------
# New: Changed-files helpers
# -------------------------

def get_monday(weeks_back: int = 0) -> datetime.date:
    """
    Return the Monday date for the current week minus `weeks_back` weeks.
    - weeks_back=0 -> this week's Monday (most recent Monday on or before today)
    - weeks_back=1 -> previous week's Monday
    """
    today = datetime.date.today()
    this_week_monday = today - datetime.timedelta(days=today.weekday())
    return this_week_monday - datetime.timedelta(weeks=weeks_back)

def changed_files_by_mtime(start_date: datetime.date,
                           end_date: datetime.date,
                           base_dir: Path,
                           exclude_fn=is_excluded) -> Dict[Path, List[Tuple[Path, datetime.datetime]]]:
    """
    Filesystem method: find files with modification time between start_date (00:00:00)
    and end_date (23:59:59). Returns mapping directory -> list of (file_path_relative_to_base, mtime).
    Excludes files for which exclude_fn(file_path) is True.
    """
    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    # Use now for end bound if end_date is today to be precise, else end of day
    if end_date == datetime.date.today():
        end_dt = datetime.datetime.now()
    else:
        end_dt = datetime.datetime.combine(end_date, datetime.time.max)

    changed = defaultdict(list)
    for f in base_dir.rglob("*"):
        if not f.is_file():
            continue
        if exclude_fn(f):
            continue
        try:
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        except Exception:
            continue
        if start_dt <= mtime <= end_dt:
            rel_dir = f.parent.relative_to(base_dir)
            rel_file = f.relative_to(base_dir)
            changed[rel_dir].append((rel_file, mtime))
    # sort file lists by mtime descending
    for k in changed:
        changed[k].sort(key=lambda x: x[1], reverse=True)
    return OrderedDict(sorted(changed.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))))

def _run_cmd(cmd: str, cwd: Path) -> Tuple[int, str, str]:
    """Run shell command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(shlex.split(cmd), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)

def changed_files_by_git(start_date: datetime.date,
                         end_date: datetime.date,
                         base_dir: Path,
                         exclude_fn=is_excluded) -> Dict[Path, List[Tuple[Path, datetime.datetime]]]:
    """
    Try to use `git` to list changed files between start_date and end_date.
    Returns mapping directory -> list of (file_path_relative_to_base, commit_datetime).
    If git is not available or the directory is not a git repo, raises RuntimeError.
    """
    # Ensure base_dir is inside a git repo
    rc, stdout, stderr = _run_cmd("git rev-parse --show-toplevel", base_dir)
    if rc != 0:
        raise RuntimeError(f"Not a git repository (or git not available): {stderr.strip()}")

    # Use ISO dates for --since/--until
    since = start_date.isoformat()
    # For until, include full day if not today; if today, use now
    if end_date == datetime.date.today():
        # git accepts e.g. "2025-10-26 23:59:59" or "now"
        until = "now"
    else:
        until = end_date.isoformat() + " 23:59:59"

    # We'll use: git log --since=<since> --until=<until> --name-only --pretty="%ci||%H"
    cmd = f'git log --since="{since}" --until="{until}" --name-only --pretty=format:"%ci||%H"'
    rc, out, err = _run_cmd(cmd, base_dir)
    if rc != 0:
        raise RuntimeError(f"git log failed: {err.strip()}")

    changed = defaultdict(list)
    current_commit_dt = None
    for line in out.splitlines():
        if not line.strip():
            continue
        if "||" in line:
            # commit header line
            # example: "2025-10-23 14:12:01 +0200||abcdef123..."
            ts_part = line.split("||", 1)[0].strip()
            try:
                # parse timezone-aware commit time
                commit_dt = datetime.datetime.strptime(ts_part, "%Y-%m-%d %H:%M:%S %z")
            except ValueError:
                # attempt without tz
                try:
                    commit_dt = datetime.datetime.strptime(ts_part, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    commit_dt = None
            current_commit_dt = commit_dt
        else:
            # filename line
            if current_commit_dt is None:
                continue
            filepath = (base_dir / line.strip()).resolve()
            if not filepath.exists():
                continue
            if exclude_fn(filepath):
                continue
            rel_dir = filepath.parent.relative_to(base_dir)
            rel_file = filepath.relative_to(base_dir)
            changed[rel_dir].append((rel_file, current_commit_dt))
    # sort and order
    for k in changed:
        changed[k].sort(key=lambda x: x[1], reverse=True)
    return OrderedDict(sorted(changed.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))))

def print_changed_dirs_only(changed_map: Dict[Path, List[Tuple[Path, datetime.datetime]]],
                            max_depth: int = 2,
                            show_total_files: bool = True):
    """
    Print only directories and subdirectories that had changes (no file-level listing).

    - changed_map: mapping relative-dir -> list of (rel_file, datetime)
    - max_depth: how many path segments to show (1 = top-level only, 2 = top + one subdir, etc.)
    - show_total_files: append total changed-file counts per directory
    """
    if not changed_map:
        print("No files changed in the requested window.\n")
        return

    # Build counts per directory and propagate counts up the tree
    dir_counts = defaultdict(int)  # Path -> int
    observed_dirs = set()

    for rel_dir, files in changed_map.items():
        # rel_dir is a Path relative to base (can be '.' for root)
        # count files present directly in that directory
        n = len(files)
        # If rel_dir is '.' treat as root Path('.')
        dirp = Path('') if str(rel_dir) == '.' else Path(rel_dir)
        observed_dirs.add(dirp)
        dir_counts[dirp] += n
        # propagate count to ancestors (so parent directories aggregate totals)
        for ancestor in dirp.parents:
            if str(ancestor) == '.':
                observed_dirs.add(Path(''))
                dir_counts[Path('')] += n
            else:
                observed_dirs.add(ancestor)
                dir_counts[ancestor] += n

    # Ensure root (.) is included if any
    if Path('') in dir_counts:
        root_total = dir_counts[Path('')]
    else:
        root_total = sum(dir_counts.values())

    # Build a sorted list of directories to display (only those observed)
    # We'll create a nested structure for pretty printing
    def parts_of(p: Path) -> Tuple[str, ...]:
        if str(p) == '.':
            return tuple()
        return tuple(p.parts)

    # Build tree nodes: dict of path tuple -> dict(children=set(), count=int)
    nodes = {}
    for d in observed_dirs:
        parts = parts_of(d)
        nodes[parts] = {"count": dir_counts.get(d, 0), "children": set()}

    # Link parent->children
    for parts in list(nodes.keys()):
        if len(parts) == 0:
            continue
        parent = parts[:-1]
        if parent in nodes:
            nodes[parent]["children"].add(parts)

    # Helper to pretty print recursively up to max_depth
    def pretty_print(node_parts: Tuple[str, ...], depth: int):
        indent = "  " * max(0, depth - 1)
        # display name
        if node_parts == tuple():
            name = "(root)"
        else:
            name = "/".join(node_parts)
        count = nodes[node_parts]["count"]
        if show_total_files:
            print(f"{indent}- {name} — {count} file(s) changed")
        else:
            print(f"{indent}- {name}")
        if depth >= max_depth:
            return
        # sort children by count desc then name
        children = sorted(nodes[node_parts]["children"],
                          key=lambda c: (-nodes[c]["count"], "/".join(c)))
        for child in children:
            pretty_print(child, depth + 1)

    # Start printing from root-level nodes (those with length 0 or 1 depending on desired view)
    print("\n===== Changed directories 02_summary (no files) =====")
    # If root present and has non-zero, show it optionally
    if nodes.get(tuple(), {}).get("count", 0) > 0:
        pretty_print(tuple(), 1)

    # print top-level directories (first segment) in stable order
    top_level_nodes = [p for p in nodes.keys() if len(p) == 1]
    top_level_nodes = sorted(top_level_nodes, key=lambda p: (-nodes[p]["count"], p[0]))

    for tl in top_level_nodes:
        pretty_print(tl, 1)

    print("=================================\n")

# -------------------------
# Run changed-files reporting
# -------------------------
if __name__ == "__main__":
    # Compute last Monday (most recent Monday on or before today)
    last_monday = get_monday(weeks_back=0)
    today = datetime.date.today()
    #print(f"Finding files changed between {last_monday.isoformat()} and {today.isoformat()} (inclusive) ...")

    # First try git (if you prefer filesystem-only, you can skip the git step)
    #try:
    #    changed_git = changed_files_by_git(last_monday, today, input_dir, exclude_fn=is_excluded)
    #    print("\n[INFO] Git-based change detection succeeded. Summary (git commits affecting files):")
    #    print_changed_dirs_only(changed_git)
    #except Exception as e:
    #    print(f"\n[INFO] Git-based detection unavailable or failed: {e}. Falling back to filesystem mtime check.")

    # Filesystem fallback (always run, useful even if git worked to show mtime-based perspective)
    changed_fs = changed_files_by_mtime(last_monday, today, input_dir, exclude_fn=is_excluded)
    #print("\n[INFO] Filesystem mtime-based change detection:")
    #print_changed_dirs_only(changed_fs)

    # Optionally: produce a compact machine-readable 02_summary (JSON) saved to disk
    out_summary = {
        "window": {"start": last_monday.isoformat(), "end": today.isoformat()},
        "by_directory": {}
    }
    for rel_dir, files in changed_fs.items():
        out_summary["by_directory"][str(rel_dir)] = [
            {"file": str(f), "mtime": (m.isoformat() if isinstance(m, datetime.datetime) else str(m))}
            for f, m in files
        ]
    #try:
    #    summary_path = input_dir / "weekly_changes_summary.json"
        #with summary_path.open("w", encoding="utf-8") as fh:
        #    json.dump(out_summary, fh, indent=2)
        #print(f"[INFO] Wrote machine-readable 02_summary to: {summary_path}")
    #except Exception as e:
    #    print(f"[WARN] Could not write 02_summary JSON: {e}")
