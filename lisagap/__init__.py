from .gap_mask_generator import GapMaskGenerator

from importlib.metadata import version

__version__ = version("lisa-gap")
__author__ = "Ollie Burke -- ollie.burke@glasgow.ac.uk"
__all__ = ["GapMaskGenerator"]

__citation__ = (
    "\n@software{lisagap,"
    "\nauthor = {Burke, Ollie and Castelli, Eleonora},"
    "\ntitle = {lisa-gap: A tool for simulating data gaps in LISA time series},"
    "\nurl = {https://github.com/ollieburke/lisa-gap},"
    f"\nversion = {{{__version__}}},"
    "\nyear = {2025}"
    "}\n"
)
