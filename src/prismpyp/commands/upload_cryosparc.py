import os
import argparse
import numpy as np
from prismpyp.utils.csparc_file_viewer import CSparcFileViewer

def add_args(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        # this script is called directly; need to create a parser
        parser = argparse.ArgumentParser(description='Generate good subset of .mrc files for re-uploading to nextPYP')
 
    else:
        # this script is called from prismpyp.__main__ entry point, in which case a parser is already created
        pass
    parser.add_argument('--intersection-folder', type=str, 
                        help='Intersection folder')
    parser.add_argument("--cs-file", type=str, 
                        help="Path to cryoSPARC job to edit's .cs file")
 
    return parser


def _normalize_name_for_match(name: str) -> str:
    """Lowercase, strip dirs and extension."""
    base = os.path.basename(str(name))
    root, _ = os.path.splitext(base)
    return root.lower()

def _load_name_set(txt_path: str) -> set[str]:
    """Load names from a .txt (one per line), stored as normalized roots."""
    out = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            root, _ = os.path.splitext(s)
            out.add(root.lower())
    return out

def _find_path_field_from_viewer(viewer: CSparcFileViewer, preferred: str | None) -> str:
    """Pick the field in the viewer that holds file paths."""
    if preferred and preferred in viewer.columns:
        return preferred
    candidates = [
        "micrograph_blob_non_dw/path",
        "micrograph_blob/path",
        "movie_blob/path",
        "exposure_blob/path",
        "blob/path",
    ]
    for c in candidates:
        if c in viewer.columns:
            return c
    for n in viewer.columns:
        if n.endswith("/path"):
            return n
    raise KeyError("Could not find a path-like field in this .cs file.")

def filter_cs_with_viewer(
    cs_path: str,
    names_txt: str,
    out_path: str,
    path_field: str | None = "micrograph_blob_non_dw/path",
    verbose: bool = True,
) -> None:
    """
    Use CSparcFileViewer to filter a cryoSPARC .cs file so that only rows whose
    basename (without extension) appear in `names_txt` are kept. Writes `out_path`.

    Matching behavior:
      - Case-insensitive
      - Ignores file extensions in both the .cs and the .txt
      - Exact root-name match (e.g., 'FoilHole_1234' matches 'FoilHole_1234.mrc')

    Parameters
    ----------
    cs_path : str
        Input .cs (NumPy structured array, possibly saved with .cs or .npy extension).
    names_txt : str
        Text file with one micrograph/movie name per line; extensions optional.
    out_path : str
        Destination path for filtered .cs (saved via np.save(..., allow_pickle=True)).
    path_field : str | None
        Column name that holds file paths. If None or not present, auto-detected.
    verbose : bool
        If True, prints a summary and a few diagnostics.
    """
    viewer = CSparcFileViewer(cs_path)
    if viewer.columns is None or len(viewer.columns) == 0:
        raise TypeError(f"{cs_path} is not a structured array with named fields.")

    path_col = _find_path_field_from_viewer(viewer, path_field)

    # Prepare the keep-list
    keep_roots = _load_name_set(names_txt)

    # Build mask
    paths = viewer[path_col]
    norm_roots = []
    mask = np.zeros(len(paths), dtype=bool)

    for i, v in enumerate(paths):
        # Handle bytes vs str robustly
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                v = v.decode(errors="ignore")
        root = _normalize_name_for_match(v)
        norm_roots.append(root)
        if any(short in root for short in keep_roots):
            mask[i] = True

    # Slice and save
    filtered = viewer.data[mask]
    
    with open(out_path, "wb") as f:
        np.save(f, filtered, allow_pickle=True)

    if verbose:
        total = len(viewer.data)
        kept = len(filtered)
        # Anything requested but not found? (Compare sets of roots)
        missing = sorted(list(keep_roots - set(norm_roots)))
        print(f"[filter_cs_with_viewer] {kept}/{total} rows kept -> {out_path}")
        if missing:
            print(f"[filter_cs_with_viewer] {len(missing)} requested names not found (showing up to 10):")
            for m in missing[:10]:
                print("  -", m)

def main(args):
    if args.cs_file is not None:
        new_cs_path = os.path.join(args.intersection_folder, "filtered.cs")
        filter_cs_with_viewer(
            cs_path=args.cs_file,
            names_txt=os.path.join(args.intersection_folder, 'files_in_common.txt'),
            out_path=new_cs_path,
            path_field=None,
            verbose=True,
        )
        
if __name__ == "__main__":
    main(add_args().parse_args())
