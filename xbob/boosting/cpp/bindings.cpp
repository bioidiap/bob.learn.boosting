#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

#include <bob/config.h>
#include <bob/python/ndarray.h>
#include <bob/python/gil.h>

#include "Machines.h"

using namespace boost::python;

static double f11(StumpMachine& s, const blitz::Array<double,1>& f){return s.forward1(f);}
static void f12(StumpMachine& s, const blitz::Array<double,2>& f, blitz::Array<double,1> p){s.forward2(f,p);}

static double f21(StumpMachine& s, const blitz::Array<uint16_t,1>& f){return s.forward1(f);}
static void f22(StumpMachine& s, const blitz::Array<uint16_t,2>& f, blitz::Array<double,1> p){s.forward2(f,p);}


static double forward1(const BoostedMachine& self, const blitz::Array<uint16_t, 1>& features){
  bob::python::no_gil t;
  return self.forward1(features);
}

static void forward2(const BoostedMachine& self, const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions, blitz::Array<double,1> labels){
  bob::python::no_gil t;
  self.forward2(features, predictions, labels);
}

static void forward3(const BoostedMachine& self, const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions, blitz::Array<double,2> labels){
  bob::python::no_gil t;
  self.forward3(features, predictions, labels);
}


static boost::shared_ptr<BoostedMachine> init_from_vector_of_weak2(object weaks, const blitz::Array<double,1>& weights){
  stl_input_iterator<boost::shared_ptr<WeakMachine> > dbegin(weaks), dend;
  boost::shared_ptr<BoostedMachine> strong = boost::make_shared<BoostedMachine>();

  for (int i = 0; dbegin != dend; ++dbegin, ++i){
    strong->add_weak_machine2(*dbegin, weights(i));
  }
  return strong;
}

static object get_weak_machines(const BoostedMachine& self){
  const std::vector<boost::shared_ptr<WeakMachine> >& weaks = self.getWeakMachines();
  list ret;
  for (std::vector<boost::shared_ptr<WeakMachine> >::const_iterator it = weaks.begin(); it != weaks.end(); ++it){
    ret.append(*it);
  }
  return ret;
}

static blitz::Array<int32_t, 1> get_indices(const BoostedMachine& self){
  return self.getIndices();
}


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
    .def("__init__", make_constructor(&init_from_vector_of_weak2, default_call_policies(), (arg("weak_classifiers"), arg("weights"))), "Uses the given list of weak classifiers and their weights.")
    .def("add_weak_machine", &BoostedMachine::add_weak_machine1, (arg("self"), arg("machine"), arg("weight")), "Adds the given weak machine with the given weight (uni-variate)")
    .def("add_weak_machine", &BoostedMachine::add_weak_machine2, (arg("self"), arg("machine"), arg("weights")), "Adds the given weak machine with the given weights (multi-variate)")
    .def("__call__", &BoostedMachine::forward1, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &BoostedMachine::forward2, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (uni-variate).")
    .def("__call__", &BoostedMachine::forward3, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (multi-variate).")
    .def("forward_p", &forward1, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("forward_p", &forward2, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (uni-variate).")
    .def("forward_p", &forward3, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (multi-variate).")
    .def("load", &BoostedMachine::load, "Reads a Machine from file")
    .def("save", &BoostedMachine::save, "Writes the machine to file")

    .def("feature_indices", &BoostedMachine::getIndices, (arg("self"), arg("start")=0, arg("end")=-1), "Get the indices required for this machine. If given, start and end limits the weak machines.")
    .add_property("indices", &get_indices, "The indices required for this machine.")
    .add_property("alpha", &BoostedMachine::getWeights, "The weights for the weak machines.")
    .add_property("weights", &BoostedMachine::getWeights, "The weights for the weak machines.")
    .add_property("weak_machines", &get_weak_machines, "The weak machines.")
  ;
}
