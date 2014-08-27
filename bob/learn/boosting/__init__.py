# import Libraries of other lib packages
import bob.io.base

# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.learn.boosting', __file__)

from .loss import *
from .trainer import *
from .machine import *

from ._library import weighted_histogram

def get_config():
  """Returns a string containing the configuration information.
  """

  import pkg_resources
  from .version import externals

  packages = pkg_resources.require(__name__)
  this = packages[0]
  deps = packages[1:]

  retval =  "%s: %s [api=0x%04x] (%s)\n" % (this.key, this.version,
      version.api, this.location)
  retval += "  - c/c++ dependencies:\n"
  for k in sorted(externals): retval += "    - %s: %s\n" % (k, externals[k])
  retval += "  - python dependencies:\n"
  for d in deps: retval += "    - %s: %s (%s)\n" % (d.key, d.version, d.location)

  return retval.strip()

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
