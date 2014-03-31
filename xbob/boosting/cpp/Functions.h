#include <bob/core/array.h>
#include "StumpMachine.h"
#include "LUTMachine.h"

inline boost::shared_ptr<WeakMachine> loadWeakMachine(bob::io::HDF5File& file){
  std::string machine_type;
  file.getAttribute(".", "MachineType", machine_type);
  if (machine_type == "LUTMachine"){
    return boost::shared_ptr<WeakMachine>(new LUTMachine(file));
  } else if (machine_type == "StumpMachine"){
    return boost::shared_ptr<WeakMachine>(new StumpMachine(file));
  }
  throw std::runtime_error("Weak machine type '" + machine_type + "' is not known or supported.");
}

