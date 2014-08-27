#include <bob/core/array.h>
#include "StumpMachine.h"
#include "LUTMachine.h"

// This is a fast implementation of the weighted histogram
inline void weighted_histogram(const blitz::Array<uint16_t,1>& features, const blitz::Array<double,1>& weights, blitz::Array<double,1>& histogram){
  assert(features.extent(0) == weights.extent(0));
  histogram = 0.;
  for (int i = features.extent(0); i--;){
    histogram((int)features(i)) += weights(i);
  }
}

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

