import scvelo as scv
import os.path as op


def read_loom(file, indir, cache=True):
    a = scv.read(op.join(indir, f"{file}.loom"), cache=cache)
    a.var_names_make_unique()
    return a
