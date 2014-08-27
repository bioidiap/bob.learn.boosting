# import the C++ stuff
#from ._boosting_old import StumpMachine, LUTMachine, BoostedMachine

from . import trainer
from . import loss
from . import machine

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]

