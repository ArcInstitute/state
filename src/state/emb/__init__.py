import sys


# Set up VCI module aliases for backward compatibility
def _setup_vci_aliases():
    """Set up vci.* aliases to point to the current emb module structure."""
    current_module = sys.modules[__name__]

    # Main vci alias
    sys.modules["vci"] = current_module

    # Import and alias submodules
    try:
        from emb import nn

        sys.modules["vci.nn"] = nn
        sys.modules["vci.nn.model"] = nn.model
    except ImportError:
        pass

    try:
        from emb import train

        sys.modules["vci.train"] = train
        sys.modules["vci.train.trainer"] = train.trainer
    except ImportError:
        pass

    try:
        from emb import data

        sys.modules["vci.data"] = data
        if hasattr(data, "loader"):
            sys.modules["vci.data.loader"] = data.loader
    except ImportError:
        pass

    try:
        from emb import utils

        sys.modules["vci.utils"] = utils
    except ImportError:
        pass

    try:
        from emb import eval as eval_module

        sys.modules["vci.eval"] = eval_module
    except ImportError:
        pass


# Set up the aliases when this module is imported
_setup_vci_aliases()

# Your existing exports
from .inference import Inference

__all__ = ["Inference"]
