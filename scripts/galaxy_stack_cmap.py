"""
This is a script for stacking signals of galaxies in a cubic map (to test projection of healpix map to cube map).
"""
import numpy as np

from galaxy_stack_hp import find_frequency_index, load_catalog

import h5py as h5


from mytools.bins import get_ids_edge


