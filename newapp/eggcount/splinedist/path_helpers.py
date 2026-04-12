import os

dirname = os.path.dirname(__file__)


def data_dir(must_exist=True, as_abs_path=False):
    """Return the absolute path to the splinedist constants directory.

    In the upstream project this returned a training-data host path with a
    fallback to a repo-relative "project/..." string. In newapp we only need
    the constants dir (which holds phi_8.npy and grid_8.npy), so this is a
    plain absolute-path lookup.
    """
    return os.path.abspath(os.path.join(dirname, "constants"))
