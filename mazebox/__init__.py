"""Single Cell Heterogeneity and Plasticity Analysis"""


# the actual API
from . import archetypes as ar
from . import plasticity as ps
from . import preprocessing as pp
from . import plotting as pl
from . import network as nw

from anndata import AnnData, concat
from anndata import (
    read_h5ad,
    read_csv,
    read_excel,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
    read_umi_tools,
)

from ._settings import settings  # start with settings as several tools are using it

set_figure_params = settings.set_figure_params


from anndata import AnnData, concat
from anndata import (
    read_h5ad,
    read_csv,
    read_excel,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
    read_umi_tools,
)
import scanpy as sc
import scvelo as scv
import os.path as op
import sys
import cellrank
