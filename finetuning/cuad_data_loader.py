"""Flat CUAD loading adapter; grouped consumers use cuad_data directly."""

try:
    from .cuad_data import DEFAULT_CHUNK_DATA_DIR as _CHUNK_DATA_DIR
    from .cuad_data import DEFAULT_RAW_DATA_DIR
    from .cuad_data import REPO_ROOT as REPO_ROOT
    from .cuad_data import filter_by_contracts as filter_by_contracts
    from .cuad_data import get_unique_contracts as get_unique_contracts
    from .cuad_data import load_cuad_data as load_cuad_data
    from .cuad_data import resolve_cuad_data_dir as resolve_cuad_data_dir
    from .cuad_data import sample_contracts as sample_contracts
except ImportError:
    from cuad_data import DEFAULT_CHUNK_DATA_DIR as _CHUNK_DATA_DIR
    from cuad_data import DEFAULT_RAW_DATA_DIR
    from cuad_data import REPO_ROOT as REPO_ROOT
    from cuad_data import filter_by_contracts as filter_by_contracts
    from cuad_data import get_unique_contracts as get_unique_contracts
    from cuad_data import load_cuad_data as load_cuad_data
    from cuad_data import resolve_cuad_data_dir as resolve_cuad_data_dir
    from cuad_data import sample_contracts as sample_contracts

DEFAULT_DATA_DIR = str(DEFAULT_RAW_DATA_DIR)
DEFAULT_CHUNK_DATA_DIR = str(_CHUNK_DATA_DIR)
