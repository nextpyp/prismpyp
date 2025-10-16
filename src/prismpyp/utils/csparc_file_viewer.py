import os
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
import numpy.lib.recfunctions as rfn
import re

class CSparcFileViewer:
    """
    Lightweight viewer for cryoSPARC .cs files (NumPy structured arrays).

    Key features:
    - Reads .cs via np.load(..., allow_pickle=True)
    - Works if the file is a single array or an .npz with one array inside
    - Presents columns (dtype names)
    - display_head() returns a small pandas DataFrame view
    - Column access options:
        * viewer["ctf/accel_kv"]           # exact column name with slash
        * viewer.ctf__accel_kv             # attribute alias: '/' -> '__'
        * viewer.get_column("ctf/accel_kv")
    """

    def __init__(self, path_to_file: str):
        self.path_to_file = path_to_file
        self.data = self._load_csparc_file()
        # Normalize to a structured array if possible
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"Loaded object is not a NumPy array: {type(self.data)}")

        if self.data.dtype.names is None:
            # Not a structured array (no named columns) — many cryoSPARC .cs ARE structured,
            # but if this one isn’t, we still store it and expose shape/values.
            self.columns = []
        else:
            self.columns = list(self.data.dtype.names)

    # -------------------------
    # Loading and validation
    # -------------------------
    def _load_csparc_file(self) -> np.ndarray:
        if not os.path.exists(self.path_to_file):
            raise FileNotFoundError(f"The file {self.path_to_file} does not exist.")

        try:
            # .cs is typically a .npy structured array saved with a custom extension.
            loaded = np.load(self.path_to_file, allow_pickle=True, mmap_mode="r")
        except Exception as e:
            raise ValueError(f"Error reading the file {self.path_to_file}: {e}")

        # Some pipelines may save as NPZ with a single member; handle that gracefully.
        if hasattr(loaded, "files"):  # np.lib.npyio.NpzFile
            if len(loaded.files) == 0:
                raise ValueError(f"{self.path_to_file} is an empty NPZ archive.")
            # Heuristics: prefer an array named 'data' if present, else the first entry.
            key = "data" if "data" in loaded.files else loaded.files[0]
            arr = loaded[key]
            loaded.close()
            return arr

        return loaded

    # -------------------------
    # Introspection
    # -------------------------
    def get_shape(self):
        return self.data.shape

    def list_columns(self, prefix: Optional[str] = None) -> Sequence[str]:
        """List all column names, optionally filtering by a 'prefix/'."""
        if not self.columns:
            return []
        if prefix is None:
            return self.columns
        pref = prefix if prefix.endswith("/") else prefix + "/"
        return [c for c in self.columns if c.startswith(pref)]

    # -------------------------
    # Viewing helpers
    # -------------------------
    def _field_subshape(self, name):
        """
        Return the subshape of a structured field, e.g., () for scalar,
        (3,) for a 3-vector, (2,2) for a small matrix field, etc.
        """
        if not self.columns or name not in self.columns:
            return ()
        # Works for structured dtypes: dtype[name].shape is the subarray shape
        return self.data.dtype[name].shape

    # --- dataframe conversion that handles non-scalar fields ---
    def to_dataframe(self, columns=None, non_scalar_mode: str = "object") -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        Parameters
        ----------
        columns : list[str] | None
            Which columns to include (default: all).
        non_scalar_mode : {"object", "flatten", "skip"}
            How to handle fields whose dtype has a subshape (e.g., (3,)):
              - "object": keep them as a single column of Python lists/tuples
              - "flatten": expand into multiple scalar columns: col[0], col[1], ...
              - "skip": drop non-scalar fields
        """
        if not self.columns:
            # Non-structured array — just wrap as-is
            return pd.DataFrame(self.data)

        if columns is None:
            columns = list(self.columns)

        out = {}
        N = len(self.data)

        for c in columns:
            subshape = self._field_subshape(c)

            # Scalar field -> pass through
            if subshape == ():
                out[c] = self.data[c]
                continue

            # Non-scalar field
            if non_scalar_mode == "object":
                # Make each cell a Python object (list), ensuring 1-D from pandas' POV
                out[c] = self.data[c].tolist()

            elif non_scalar_mode == "skip":
                # Ignore this field entirely
                continue

            elif non_scalar_mode == "flatten":
                # Flatten last dimensions into k scalar columns
                k = int(np.prod(subshape))
                flat = self.data[c].reshape(N, k)
                for i in range(k):
                    out[f"{c}[{i}]"] = flat[:, i]

            else:
                raise ValueError("non_scalar_mode must be one of {'object','flatten','skip'}")

        return pd.DataFrame(out)

    def display_head(self, n: int = 5, non_scalar_mode: str = "object") -> pd.DataFrame:
        """
        Return first n rows as a DataFrame. By default, non-scalar fields
        become object columns (lists/tuples) so pandas doesn't choke.
        """
        return self.to_dataframe(non_scalar_mode=non_scalar_mode).head(n)

    # -------------------------
    # Column access
    # -------------------------
    @staticmethod
    def _alias_to_real(name: str) -> str:
        """
        Map attribute-friendly alias to real column name.
        Convention: '/' is represented as '__' (double underscore) in attribute access.
        Example: 'ctf__accel_kv' -> 'ctf/accel_kv'
        """
        return name.replace("__", "/")

    def get_column(self, name: str) -> np.ndarray:
        """Get a column by exact name (with slashes), e.g. 'ctf/accel_kv'."""
        if not self.columns:
            raise AttributeError("This .cs file does not have named columns (not a structured array).")
        if name not in self.columns:
            raise AttributeError(f"Column '{name}' not found. Available columns include: {self.columns[:10]}{' ...' if len(self.columns) > 10 else ''}")
        return self.data[name]

    def add_trimmed_path_column(self, new_col_name: str = "trimmed_path", keep_extension: bool = True) -> np.ndarray:
        """
        Create a cleaned path column by removing the '<uid>_' prefix from the path,
        and (optionally) remove the file extension. Appends it as `new_col_name`.

        Parameters
        ----------
        new_col_name : str
            Name of the new column (default: "trimmed_path")
        remove_extension : bool
            If True, strip known cryo-EM extensions like .mrc, .mrcs, .tif, .tiff, etc.
        """
        if not self.columns:
            raise TypeError("This file has no named columns (not a structured array).")

        # Resolve source columns (using your alias scheme if applicable)
        uid_col = "uid"
        if uid_col not in self.columns:
            raise KeyError("Column 'uid' not found.")

        path_col = (
            "micrograph_blob_non_dw/path"
            if "micrograph_blob_non_dw/path" in self.columns
            else "ctf/path" if "ctf/path" in self.columns
            else None
        )
        if path_col is None:
            raise KeyError("Neither 'micrograph_blob_non_dw/path' nor 'ctf/path' found.")

        # Prepare Python string lists
        uids = self.data[uid_col]
        paths = self.data[path_col]

        def _as_str(x):
            # bytes -> str; None/np.nan safe-ish; everything else via str()
            if x is None:
                return ""
            if isinstance(x, bytes):
                try:
                    return x.decode("utf-8", errors="ignore")
                except Exception:
                    return x.decode(errors="ignore")
            return str(x)

        trimmed = []
        ext_pattern = re.compile(r"\.(mrcs?|tif{1,2}|png|jpg|jpeg|eer|dat|txt)$", re.IGNORECASE)
        
        for u, p in zip(uids, paths):
            p_str = _as_str(p)
            u_str = f"{_as_str(u)}_"
            if u_str and u_str in p_str:
                cleaned = p_str.split(u_str, 1)[-1].lstrip("/")
            else:
                cleaned = p_str
                
            # Optionally remove file extension
            if not keep_extension:
                cleaned = ext_pattern.sub("", cleaned)
            trimmed.append(cleaned)

        # Choose a fixed-width Unicode dtype big enough for the longest value
        maxlen = max(1, max((len(s) for s in trimmed), default=1))
        new_dtype = f"<U{maxlen}"

        # If column already exists, replace it; else append it.
        if new_col_name in self.columns:
            # Replace existing field values without changing dtype layout
            # If dtype width is too small, re-append with larger dtype
            if self.data.dtype[new_col_name].kind == "U" and self.data.dtype[new_col_name].itemsize // 4 >= maxlen:
                self.data[new_col_name] = np.array(trimmed, dtype=self.data.dtype[new_col_name])
            else:
                # Rebuild array with widened column
                without = [c for c in self.columns if c != new_col_name]
                base = self.data[without]
                self.data = rfn.append_fields(base, new_col_name, np.array(trimmed, dtype=new_dtype), usemask=False)
        else:
            # Append new field
            self.data = rfn.append_fields(self.data, new_col_name, np.array(trimmed, dtype=new_dtype), usemask=False)
            self.columns.append(new_col_name)

        return self.data[new_col_name]
    
    # Dictionary-style access: viewer["ctf/accel_kv"]
    def __getitem__(self, key: Union[str, Sequence[str]]) -> Union[np.ndarray, np.void, np.recarray]:
        if isinstance(key, str):
            return self.get_column(key)
        # Allow selecting multiple columns: viewer[["ctf/accel_kv", "ctf/defocus_u"]]
        if isinstance(key, (list, tuple)):
            missing = [k for k in key if k not in self.columns]
            if missing:
                raise KeyError(f"Columns not found: {missing}")
            return self.data[key]
        raise TypeError("Key must be a string column name or a sequence of names.")

    # Attribute-style access: viewer.ctf__accel_kv  (alias for 'ctf/accel_kv')
    def __getattr__(self, name: str):
        # Only invoked if normal attributes aren't found.
        real = self._alias_to_real(name)
        if self.columns and real in self.columns:
            return self.data[real]
        # Fall back to default behavior
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # Nice repr
    def __repr__(self):
        shape = self.get_shape()
        col_info = f"{len(self.columns)} columns" if self.columns else "no named columns"
        return f"<CSparcFileViewer shape={shape}, {col_info}, path='{self.path_to_file}'>"