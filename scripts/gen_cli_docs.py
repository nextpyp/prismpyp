#!/usr/bin/env python3
"""
Generate Markdown pages from `prismpyp ... --help` for MkDocs.
Writes files to docs/cli/*.md
"""
import subprocess
from pathlib import Path

# List every subcommand you want documented ("" = top-level help)
COMMANDS = [
    "",                  # prismpyp --help
    "train",
    "eval2d",
    "eval3d",
    "metadata_nextpyp",
    "metadata_cryosparc",
    "intersect",
    "visualizer",
]

OUT_DIR = Path("docs/cli")

def run_help(cmd_parts):
    try:
        out = subprocess.check_output(cmd_parts, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        # Some CLIs exit nonzero on --help; still capture output
        out = e.output or ""
    return out

def write_md(name, content):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    title = "prismpyp" if not name else f"prismpyp {name}"
    fname = OUT_DIR / ("prismpyp.md" if not name else f"{name}.md")
    lines = []
    lines.append(f"# `{title} --help`")
    lines.append("")
    lines.append("```bash")
    lines.append(f"$ {title} --help")
    lines.append(content.rstrip())
    lines.append("```")
    fname.write_text("\n".join(lines), encoding="utf-8")

def main():
    for name in COMMANDS:
        cmd = ["prismpyp"]
        if name:
            cmd.append(name)
        cmd.append("--help")
        help_text = run_help(cmd)
        write_md(name, help_text)

if __name__ == "__main__":
    main()
