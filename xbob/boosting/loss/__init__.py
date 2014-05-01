from . import LossFunction # Just to get the documentation for it
from .ExponentialLoss import ExponentialLoss
from .LogitLoss import LogitLoss
from .TangentialLoss import TangentialLoss
# from .JesorskyLoss import JesorskyLoss
from .._library import LossFunction, JesorskyLoss

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
