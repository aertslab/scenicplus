# UV Setup for SCENIC+

This project is now configured to use `uv`, a fast Python package installer and resolver.

## Prerequisites

1. **Python Version**: This project currently requires **Python 3.11**. Python 3.12 is not yet supported due to the `sinfo` package incompatibility (used by `scanpy`).

2. Install HDF5 (required for tables package):
   
   **macOS:**
   ```bash
   brew install hdf5
   ```
   
   **Linux:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libhdf5-dev
   
   # Fedora/RHEL
   sudo dnf install hdf5-devel
   ```
   
   **Windows:**
   
   On Windows, the pre-built wheels for `tables` 3.10+ should work without manually installing HDF5. If you encounter issues, you can install HDF5 via conda:
   ```bash
   conda install -c conda-forge hdf5
   ```

## Installation

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

The project includes a dependency override for the `tables` package to use version 3.10+ instead of the problematic 3.9.2, avoiding compilation issues on macOS.

## Running Python with uv

Use `uv run` to execute Python commands in the managed environment:

```bash
uv run python your_script.py
uv run python -c "import scenicplus"
```

## How It Works

The `pyproject.toml` file includes a `[tool.uv]` section with an override:

```toml
[tool.uv]
override-dependencies = [
    "tables>=3.10"
]
```

This ensures that even though some dependencies (like `scanpy`) pin `tables==3.9.2`, uv will use version 3.10+ which has pre-built wheels for Apple Silicon and avoids compilation errors.

## Troubleshooting

### tables Compilation Errors

If you see compilation errors mentioning `fdopen` or `zutil.c`, the `tables` package is trying to build from source. This should not happen with the current setup, but if it does:

1. Ensure your `pyproject.toml` has the override-dependencies section shown above
2. Delete `uv.lock` and regenerate:
   ```bash
   rm uv.lock
   uv lock
   uv sync
   ```

### Verifying Installation

Check that tables 3.10+ is installed:
```bash
uv run python -c "import tables; print(tables.__version__)"
```

You should see version 3.10.2 or later.

## Development

The project uses:
- Python 3.11 (3.12 support blocked by `sinfo` package used by `scanpy`)
- uv for package management and virtual environment
- All dependencies specified in `requirements.txt`
- Dependency overrides in `pyproject.toml` for compatibility:
  - `tables>=3.10` (fixes macOS compilation issues)
  - `ray>=2.51.0` (adds Python 3.12 wheel support, ready for when Python 3.12 is supported)

### Python 3.12 Status

Python 3.12 support is currently blocked by the `sinfo` package (v0.3.4), which uses `configparser.SafeConfigParser` (removed in Python 3.12). This package is a dependency of `scanpy` v1.8.2 and is only used for the optional `sc.logging.print_versions()` function. Once `scanpy` updates or removes this dependency, Python 3.12 will be supported.
