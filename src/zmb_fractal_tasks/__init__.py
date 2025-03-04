"""
A collection of fractal tasks used for varying ZMB projects.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zmb-fractal-tasks")
except PackageNotFoundError:
    __version__ = "uninstalled"