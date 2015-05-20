# import Libraries of other lib packages
import bob.io.base

# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.learn.boosting', __file__)

# versioning
from . import version
from .version import module as __version__
from .version import api as __api_version__

# include loss functions
from . import LossFunction # Just to get the documentation for it
from .ExponentialLoss import ExponentialLoss
from .LogitLoss import LogitLoss
from .TangentialLoss import TangentialLoss
from ._library import JesorskyLoss

# include trainers
from .StumpTrainer import StumpTrainer
from .Boosting import Boosting
from ._library import LUTTrainer

# include machines
from ._library import WeakMachine, StumpMachine, LUTMachine, BoostedMachine

# include auxiliary functions
from ._library import weighted_histogram

def get_config():
  """Returns a string containing the configuration information.
  """
  return bob.extension.get_config(__name__, version.externals, version.api)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
