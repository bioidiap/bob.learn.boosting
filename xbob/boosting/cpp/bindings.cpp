#include <boost/python.hpp>

#include <bob/config.h>
#include <bob/python/ndarray.h>

#include "Machines.h"

using namespace boost::python;

static double f11(StumpMachine& s, const blitz::Array<double,1>& f){return s.forward1(f);}
static void f12(StumpMachine& s, const blitz::Array<double,2>& f, blitz::Array<double,1> p){s.forward2(f,p);}

static double f21(StumpMachine& s, const blitz::Array<uint16_t,1>& f){return s.forward1(f);}
static void f22(StumpMachine& s, const blitz::Array<uint16_t,2>& f, blitz::Array<double,1> p){s.forward2(f,p);}

BOOST_PYTHON_MODULE(_boosting) {
  bob::python::setup_python("Bindings for the xbob.boosting machines.");

  class_<WeakMachine, boost::shared_ptr<WeakMachine>, boost::noncopyable>("WeakMachine", "Pure virtual base class for weak machines", no_init);

  class_<StumpMachine, boost::shared_ptr<StumpMachine>, bases<WeakMachine> >("StumpMachine", "A machine comparing features to a threshold.", no_init)
    .def(init<double, double, int >((arg("self"), arg("threshold"), arg("polarity"), arg("index")), "Creates a StumpMachine with the given threshold, polarity and the feature index, for which the machine is valid."))
    .def(init<bob::io::HDF5File&>((arg("self"),arg("file")), "Creates a new machine from file."))
    .def("__call__", &f11, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &f12, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate only).")
    .def("__call__", &f21, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &f22, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate only).")
    .def("load", &StumpMachine::load, "Reads a Machine from file")
    .def("save", &StumpMachine::save, "Writes the machine to file")

    .def("feature_indices", &StumpMachine::getIndices, "The indices into the feature vector required by this machine.")
    .add_property("threshold", &StumpMachine::getThreshold, "The threshold of this machine.")
    .add_property("polarity", &StumpMachine::getPolarity, "The polarity for this machine.")
  ;

  class_<LUTMachine, boost::shared_ptr<LUTMachine>, bases<WeakMachine> >("LUTMachine", "A machine containing a Look-Up-Table.", no_init)
    .def(init<const blitz::Array<double,2>&, const blitz::Array<int,1>&>((arg("self"), arg("look_up_tables"), arg("indices")), "Creates a LUTMachine with the given look-up-table and the feature indices, for which the LUT is valid."))
    .def(init<bob::io::HDF5File&>((arg("self"),arg("file")), "Creates a new machine from file."))
    .def("__call__", &LUTMachine::forward1, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &LUTMachine::forward2, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate).")
    .def("__call__", &LUTMachine::forward3, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (multi-variate).")
    .def("load", &LUTMachine::load, "Reads a Machine from file")
    .def("save", &LUTMachine::save, "Writes the machine to file")

    .add_property("lut", &LUTMachine::getLut, "The look up table of the machine.")
    .def("feature_indices", &LUTMachine::getIndices, "The indices into the feature vector required by this machine.")
  ;

  class_<BoostedMachine, boost::shared_ptr<BoostedMachine> >("BoostedMachine",  "A machine containing of several weak machines", no_init)
    .def(init<>(arg("self"), "Creates an empty machine."))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("file")), "Creates a new machine from file"))
    .def("add_weak_machine", &BoostedMachine::add_weak_machine1, (arg("self"), arg("machine"), arg("weight")), "Adds the given weak machine with the given weight (uni-variate)")
    .def("add_weak_machine", &BoostedMachine::add_weak_machine2, (arg("self"), arg("machine"), arg("weights")), "Adds the given weak machine with the given weights (multi-variate)")
    .def("__call__", &BoostedMachine::forward1, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &BoostedMachine::forward2, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (uni-variate).")
    .def("__call__", &BoostedMachine::forward3, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (multi-variate).")
    .def("load", &BoostedMachine::load, "Reads a Machine from file")
    .def("save", &BoostedMachine::save, "Writes the machine to file")

    .def("feature_indices", &BoostedMachine::getIndices, (arg("self")), "Returns the indices required for this machine.")
    .def("alpha", &BoostedMachine::getWeights, (arg("self")), "Returns the weights for the weak machines.")
  ;
}
