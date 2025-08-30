# Documentation

This directory contains the Sphinx documentation for lisa-gap.

## Building Documentation Locally

1. Install documentation dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

2. Build the documentation:
   ```bash
   cd docs
   make html
   ```

3. View the documentation:
   Open `build/html/index.html` in your browser.

## Read the Docs

The documentation is automatically built and hosted on Read the Docs when changes are pushed to the repository.

## Structure

- `source/` - Documentation source files
- `source/conf.py` - Sphinx configuration
- `source/gap_notebook.ipynb` - Tutorial notebook (copied from root)
- `requirements.txt` - Dependencies for Read the Docs
- `Makefile` - Build commands for Unix systems
- `make.bat` - Build commands for Windows

## Adding Content for development purposes

To add new pages:

1. Create a new `.rst` file in `source/`
2. Add it to the `toctree` in `source/index.rst`
3. Rebuild the documentation
