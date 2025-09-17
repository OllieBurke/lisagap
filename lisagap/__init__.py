from .gap_window_generator import GapWindowGenerator
from .gap_segment_generator import DataSegmentGenerator

# Import GapMaskGenerator from lisaglitch
try:
    from lisaglitch import GapMaskGenerator
except ImportError:
    raise ImportError(
        "lisaglitch is required but not installed. "
        "Please install it with: pip install lisaglitch"
    )

from importlib.metadata import version

__version__ = version("lisa-gap")
__author__ = "Ollie Burke -- ollie.burke@glasgow.ac.uk"
__all__ = ["GapWindowGenerator", "DataSegmentGenerator", "GapMaskGenerator"]

__citation__ = (
    "\n@software{lisagap,"
    "\nauthor = {Burke, Ollie and Castelli, Eleonora},"
    "\ntitle = {lisa-gap: A tool for simulating data gaps in LISA time series},"
    "\nurl = {https://github.com/ollieburke/lisa-gap},"
    f"\nversion = {{{__version__}}},"
    "\nyear = {2025}"
    "}\n"
)
