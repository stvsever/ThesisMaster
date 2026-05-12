#!/usr/bin/env python3
"""
Thesis Word document build system.

Like tectonic for LaTeX, but for Word:
  - `python build.py unpack`   → explode masterproef.docx into editable XML in unpacked/
  - `python build.py pack`     → repack unpacked/ back to masterproef.docx
  - `python build.py rebuild`  → unpack → pack (round-trip, preserves exact content)
  - `python build.py figures`  → sync assets/figures into the unpacked media/ folder
  - `python build.py validate` → validate unpacked/ without packing

Editing workflow (equivalent to editing a .tex file and running tectonic):
  1. python build.py unpack
  2. Edit XML in unpacked/word/document.xml (or other XML files)
     OR replace figures in assets/figures/main/ and run `python build.py figures`
  3. python build.py pack

Agent workflow:
  - Agents can call `python build.py unpack`, make targeted XML edits, then `python build.py pack`.
  - All figures live in:  ../assets/figures/main/fig_NN.png  (main)
                          ../assets/figures/sup/fig_sNN.png  (supplementary)
  - All tables live in:   ../assets/tables/main/             (main)
                          ../assets/tables/sup/              (supplementary)
  - Template logos live in: ../assets/template/
"""

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

DOCX        = os.path.join(SCRIPT_DIR, 'masterproef.docx')
UNPACKED    = os.path.join(SCRIPT_DIR, 'unpacked')
SCRIPTS_DIR = os.path.join(SCRIPT_DIR, 'scripts', 'office')

ASSETS_FIGURES_MAIN = os.path.join(REPO_ROOT, 'assets', 'figures', 'main')
ASSETS_FIGURES_SUP  = os.path.join(REPO_ROOT, 'assets', 'figures', 'sup')
ASSETS_TABLES_MAIN  = os.path.join(REPO_ROOT, 'assets', 'tables', 'main')
ASSETS_TABLES_SUP   = os.path.join(REPO_ROOT, 'assets', 'tables', 'sup')
ASSETS_TEMPLATE     = os.path.join(REPO_ROOT, 'assets', 'template')

UNPACK_SCRIPT   = os.path.join(SCRIPTS_DIR, 'unpack.py')
PACK_SCRIPT     = os.path.join(SCRIPTS_DIR, 'pack.py')
VALIDATE_SCRIPT = os.path.join(SCRIPTS_DIR, 'validate.py')


def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def cmd_unpack(args):
    """Unpack masterproef.docx → unpacked/"""
    if os.path.exists(UNPACKED):
        print(f"Removing existing {UNPACKED}/")
        shutil.rmtree(UNPACKED)
    run(['python3', UNPACK_SCRIPT, DOCX, UNPACKED])
    print(f"\nUnpacked to {UNPACKED}/")
    print("Edit unpacked/word/document.xml then run: python build.py pack")


def cmd_pack(args):
    """Repack unpacked/ → masterproef.docx"""
    if not os.path.exists(UNPACKED):
        sys.exit(f"ERROR: {UNPACKED}/ not found. Run `python build.py unpack` first.")
    run(['python3', PACK_SCRIPT, UNPACKED, DOCX, '--original', DOCX])
    print(f"\nPacked to {DOCX}")
    if not args.keep:
        shutil.rmtree(UNPACKED)
        print(f"Cleaned up {UNPACKED}/")


def cmd_rebuild(args):
    """Round-trip: unpack then pack (useful for normalizing XML)"""
    cmd_unpack(args)
    args.keep = False
    cmd_pack(args)


def cmd_figures(args):
    """
    Sync figures from assets/ into the unpacked media folder.

    Reads the document XML to find which rId maps to which figure,
    then replaces the media file in unpacked/word/media/ with the
    versioned file from assets/figures/.

    Must call `python build.py unpack` first.
    """
    if not os.path.exists(UNPACKED):
        sys.exit(f"ERROR: {UNPACKED}/ not found. Run `python build.py unpack` first.")

    media_dir = os.path.join(UNPACKED, 'word', 'media')
    rels_path = os.path.join(UNPACKED, 'word', '_rels', 'document.xml.rels')

    # Build rId → filename map
    rels_tree = ET.parse(rels_path)
    rels = {}
    for r in rels_tree.getroot():
        rels[r.get('Id')] = r.get('Target')

    # Find embed order in document
    import re
    doc_xml = open(os.path.join(UNPACKED, 'word', 'document.xml')).read()
    embeds = re.findall(r'r:embed="(rId\d+)"', doc_xml)

    print(f"Found {len(embeds)} embedded images in document order:")
    for seq, rid in enumerate(embeds, start=1):
        target = rels.get(rid, '?')
        media_file = os.path.basename(target)
        src_name = f"fig_{seq:02d}.png"
        src_path = os.path.join(ASSETS_FIGURES_MAIN, src_name)
        dst_path = os.path.join(media_dir, media_file)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Fig {seq:02d}: {src_name} → media/{media_file}  [updated]")
        else:
            print(f"  Fig {seq:02d}: {src_name} → media/{media_file}  [no source, kept existing]")

    print("\nFigures synced. Run `python build.py pack` to rebuild the docx.")


def cmd_validate(args):
    """Validate the current unpacked/ directory without packing."""
    if not os.path.exists(UNPACKED):
        sys.exit(f"ERROR: {UNPACKED}/ not found. Run `python build.py unpack` first.")
    run(['python3', VALIDATE_SCRIPT, UNPACKED, '--original', DOCX])


def cmd_status(args):
    """Show current state of the working tree."""
    docx_exists    = os.path.exists(DOCX)
    unpacked_exists = os.path.exists(UNPACKED)
    figs_main = sorted(os.listdir(ASSETS_FIGURES_MAIN)) if os.path.exists(ASSETS_FIGURES_MAIN) else []
    figs_sup  = sorted(os.listdir(ASSETS_FIGURES_SUP))  if os.path.exists(ASSETS_FIGURES_SUP)  else []

    print(f"masterproef.docx : {'exists' if docx_exists else 'MISSING'}")
    print(f"unpacked/        : {'exists (unpack in progress)' if unpacked_exists else 'not present'}")
    print(f"figures/main/    : {len(figs_main)} files  ({', '.join(figs_main[:5])}{'...' if len(figs_main) > 5 else ''})")
    print(f"figures/sup/     : {len(figs_sup)} files")


def main():
    parser = argparse.ArgumentParser(
        description='Thesis .docx build system (like tectonic for Word)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('unpack',   help='Explode docx into editable XML in unpacked/')
    p_pack = sub.add_parser('pack', help='Repack unpacked/ back to masterproef.docx')
    p_pack.add_argument('--keep', action='store_true', help='Keep unpacked/ after packing')
    sub.add_parser('rebuild',  help='Unpack then pack (round-trip / normalize)')
    sub.add_parser('figures',  help='Sync assets/figures into unpacked/word/media/')
    sub.add_parser('validate', help='Validate unpacked/ without packing')
    sub.add_parser('status',   help='Show current build state')

    args = parser.parse_args()

    dispatch = {
        'unpack':   cmd_unpack,
        'pack':     cmd_pack,
        'rebuild':  cmd_rebuild,
        'figures':  cmd_figures,
        'validate': cmd_validate,
        'status':   cmd_status,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
