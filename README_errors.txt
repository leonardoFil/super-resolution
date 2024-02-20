After env creation, got this error:
    from . import ft2font
ImportError: DLL load failed: The specified module could not be found.

Resolved by:
pip uninstall matplotlib
pip install --upgrade matplotlib