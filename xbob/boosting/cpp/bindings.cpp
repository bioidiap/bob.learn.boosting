#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <boost/python/stl_iterator.hpp>

#include <bob/config.h>
#include <bob/python/ndarray.h>
#include <bob/python/gil.h>

#include "WeakMachine.h"
#include "StumpMachine.h"
#include "LUTMachine.h"
#include "BoostedMachine.h"

#include "JesorskyLoss.h"

#include "LUTTrainer.h"
#include "Functions.h"

using namespace boost::python;

// Stump machine access
static double f11(StumpMachine& s, const blitz::Array<double,1>& f){return s.forward1(f);}
static void f12(StumpMachine& s, const blitz::Array<double,2>& f, blitz::Array<double,1> p){s.forward3(f,p);}
static void f13(StumpMachine& s, const blitz::Array<double,2>& f, blitz::Array<double,2> p){s.forward4(f,p);}

static double f21(StumpMachine& s, const blitz::Array<uint16_t,1>& f){return s.forward1(f);}
static void f22(StumpMachine& s, const blitz::Array<uint16_t,2>& f, blitz::Array<double,1> p){s.forward3(f,p);}
static void f23(StumpMachine& s, const blitz::Array<uint16_t,2>& f, blitz::Array<double,2> p){s.forward4(f,p);}

// boosted machine access, which allows multi-threading
static double forward1(const BoostedMachine& self, const blitz::Array<uint16_t, 1>& features){bob::python::no_gil t; return self.forward1(features);}
static void forward2(const BoostedMachine& self, const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions){bob::python::no_gil t; self.forward2(features, predictions);}

static void forward3(const BoostedMachine& self, const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions){bob::python::no_gil t; self.forward3(features, predictions);}
static void forward4(const BoostedMachine& self, const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions){bob::python::no_gil t; self.forward4(features, predictions);}

static void forward5(const BoostedMachine& self, const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions, blitz::Array<double,1> labels){bob::python::no_gil t; self.forward5(features, predictions, labels);}
static void forward6(const BoostedMachine& self, const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions, blitz::Array<double,2> labels){bob::python::no_gil t; self.forward6(features, predictions, labels);}

static boost::shared_ptr<BoostedMachine> init_from_vector_of_weak2(object weaks, const blitz::Array<double,1>& weights){
  stl_input_iterator<boost::shared_ptr<WeakMachine> > it(weaks), endIt;
  boost::shared_ptr<BoostedMachine> strong = boost::make_shared<BoostedMachine>();

  for (int i = 0; it != endIt; ++it, ++i){
    strong->add_weak_machine1(*it, weights(i));
  }
  return strong;
}

// Wrapper function to translate std::vector of WeakMachines (C++) into list of WeakMachines (Python)
static object get_weak_machines(const BoostedMachine& self){
  const std::vector<boost::shared_ptr<WeakMachine> >& weaks = self.getWeakMachines();
  list ret;
  for (std::vector<boost::shared_ptr<WeakMachine> >::const_iterator it = weaks.begin(); it != weaks.end(); ++it){
    ret.append(*it);
  }
  return ret;
}

// Wrapper function to have get_indices as a property of the Python object
static blitz::Array<int32_t, 1> get_indices(const BoostedMachine& self){
  return self.getIndices();
}

#if 0
// Wrapper functions for the weighted histogram
static void weighted_histogram1(bob::python::const_ndarray features, bob::python::const_ndarray weights, blitz::Array<double,1> histogram){
  weighted_histogram(features.bz<uint16_t,1>(), weights.bz<double,1>(), histogram);
}
static blitz::Array<double, 1> weighted_histogram2(bob::python::const_ndarray features, bob::python::const_ndarray weights, const uint16_t bin_count){
  blitz::Array<double,1> retval(bin_count);
  weighted_histogram(features.bz<uint16_t,1>(), weights.bz<double,1>(), retval);
  return retval;
}
#endif

// Wrapper functions for Jesorsky loss
static blitz::Array<double,1> jesorsky_loss_sum(const JesorskyLoss& loss, const blitz::Array<double,1>& alpha, const blitz::Array<double,2>& targets, const blitz::Array<double,2>& previous_scores, const blitz::Array<double,2>& current_scores){
  blitz::Array<double, 1> retval(1);
  loss.lossSum(alpha, targets, previous_scores, current_scores, retval);
  return retval;
}

static blitz::Array<double,1> jesorsky_gradient_sum(const JesorskyLoss& loss, const blitz::Array<double,1>& alpha, const blitz::Array<double,2>& targets, const blitz::Array<double,2>& previous_scores, const blitz::Array<double,2>& current_scores){
  blitz::Array<double, 1> retval(targets.extent(1));
  loss.gradientSum(alpha, targets, previous_scores, current_scores, retval);
  return retval;
}


static blitz::Array<double,2> jesorsky_loss(const JesorskyLoss& loss, const blitz::Array<double,2>& targets, const blitz::Array<double, 2>& scores){
  blitz::Array<double,2> retval(targets.extent(0), 1);
  loss.loss(targets, scores, retval);
  return retval;
}

static blitz::Array<double,2> jesorsky_loss_gradient(const JesorskyLoss& loss, const blitz::Array<double,2>& targets, const blitz::Array<double, 2>& scores){
  blitz::Array<double,2> retval(targets.extent(0), targets.extent(1));
  loss.lossGradient(targets, scores, retval);
  return retval;
}


BOOST_PYTHON_MODULE(_boosting) {
  bob::python::setup_python("Bindings for the xbob.boosting machines.");

  // bind the weak machine
  class_<WeakMachine, boost::shared_ptr<WeakMachine>, boost::noncopyable>("WeakMachine", "Pure virtual base class for weak machines", no_init);

  // bind the decision stump classifier
  class_<StumpMachine, boost::shared_ptr<StumpMachine>, bases<WeakMachine> >("StumpMachine", "A machine comparing features to a threshold.", no_init)
    .def(init<double, double, int >((arg("self"), arg("threshold"), arg("polarity"), arg("index")), "Creates a StumpMachine with the given threshold, polarity and the feature index, for which the machine is valid."))
    .def(init<bob::io::HDF5File&>((arg("self"),arg("file")), "Creates a new machine from file."))
    .def("__call__", &f11, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &f12, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate only).")
    .def("__call__", &f13, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate only).")
    .def("__call__", &f21, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &f22, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate only).")
    .def("__call__", &f23, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate only).")
    .def("load", &StumpMachine::load, "Reads a Machine from file")
    .def("save", &StumpMachine::save, "Writes the machine to file")

    .def("feature_indices", &StumpMachine::getIndices, "The indices into the feature vector required by this machine.")
    .add_property("threshold", &StumpMachine::getThreshold, "The threshold of this machine.")
    .add_property("polarity", &StumpMachine::getPolarity, "The polarity for this machine.")
  ;

  // bind the look-up-table classifier
  class_<LUTMachine, boost::shared_ptr<LUTMachine>, bases<WeakMachine> >("LUTMachine", "A machine containing a Look-Up-Table.", no_init)
    .def(init<const blitz::Array<double,1>&, const int>((arg("self"), arg("look_up_table"), arg("index")), "Creates a LUTMachine with the given look-up-table and the feature index, for which the LUT is valid (uni-variate case)."))
    .def(init<const blitz::Array<double,2>&, const blitz::Array<int,1>&>((arg("self"), arg("look_up_tables"), arg("indices")), "Creates a LUTMachine with the given look-up-table and the feature indices, for which the LUT is valid (multi-variate case)."))
    .def(init<bob::io::HDF5File&>((arg("self"),arg("file")), "Creates a new machine from file."))
    .def("__call__", &LUTMachine::forward1, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &LUTMachine::forward2, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature (multi-variate).")
    .def("__call__", &LUTMachine::forward3, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate).")
    .def("__call__", &LUTMachine::forward4, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (multi-variate).")
    .def("load", &LUTMachine::load, "Reads a Machine from file")
    .def("save", &LUTMachine::save, "Writes the machine to file")

    .add_property("lut", &LUTMachine::getLut, "The look up table of the machine.")
    .def("feature_indices", &LUTMachine::getIndices, "The indices into the feature vector required by this machine.")
  ;

  // bind the boosted machine
  class_<BoostedMachine, boost::shared_ptr<BoostedMachine> >("BoostedMachine",  "A machine containing of several weak machines", no_init)
    .def(init<>(arg("self"), "Creates an empty machine."))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("file")), "Creates a new machine from file"))
    .def("__init__", make_constructor(&init_from_vector_of_weak2, default_call_policies(), (arg("weak_classifiers"), arg("weights"))), "Uses the given list of weak classifiers and their weights.")
    .def("add_weak_machine", &BoostedMachine::add_weak_machine1, (arg("self"), arg("machine"), arg("weight")), "Adds the given weak machine with the given weight (uni-variate)")
    .def("add_weak_machine", &BoostedMachine::add_weak_machine2, (arg("self"), arg("machine"), arg("weights")), "Adds the given weak machine with the given weights (multi-variate)")
    .def("__call__", &BoostedMachine::forward1, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("__call__", &BoostedMachine::forward2, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature vector (multi-variate).")
    .def("__call__", &BoostedMachine::forward3, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate).")
    .def("__call__", &BoostedMachine::forward4, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (multi-variate).")
    .def("__call__", &BoostedMachine::forward5, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (uni-variate).")
    .def("__call__", &BoostedMachine::forward6, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (multi-variate).")
    .def("forward_p", &forward1, (arg("self"), arg("features")), "Returns the prediction for the given feature vector.")
    .def("forward_p", &forward2, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature vector (multi-variate).")
    .def("forward_p", &forward3, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (uni-variate).")
    .def("forward_p", &forward4, (arg("self"), arg("features"), arg("predictions")), "Computes the predictions for the given feature set (multi-variate).")
    .def("forward_p", &forward5, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (uni-variate).")
    .def("forward_p", &forward6, (arg("self"), arg("features"), arg("predictions"), arg("labels")), "Computes the predictions and the labels for the given feature set (multi-variate).")
    .def("load", &BoostedMachine::load, "Reads a Machine from file")
    .def("save", &BoostedMachine::save, "Writes the machine to file")

    .def("feature_indices", &BoostedMachine::getIndices, (arg("self"), arg("start")=0, arg("end")=-1), "Get the indices required for this machine. If given, start and end limits the weak machines.")
    .add_property("indices", &get_indices, "The indices required for this machine.")
    .add_property("alpha", &BoostedMachine::getWeights, "The weights for the weak machines.")
    .add_property("weights", &BoostedMachine::getWeights, "The weights for the weak machines.")
    .add_property("outputs", &BoostedMachine::numberOfOutputs, "The number of outputs of the multi-variate classifier (1 in case of uni-variate classifier).")
    .add_property("weak_machines", &get_weak_machines, "The weak machines.")
  ;

  enum_<LUTTrainer::SelectionStyle>("SelectionStyle")
    .value("independent", LUTTrainer::independent)
    .value("shared", LUTTrainer::shared)
    .export_values();

  class_<LUTTrainer,  boost::shared_ptr<LUTTrainer> >("LUTTrainer",  "A trainer to train a LUTMachine", init<uint16_t, optional<int,LUTTrainer::SelectionStyle> >())
    .def("train", &LUTTrainer::train, (arg("self"), arg("training_features"), arg("loss_gradient")), "Trains and returns a LUTMachine.")
    .add_property("number_of_labels", &LUTTrainer::maximumFeatureValue, "The highest feature value + 1.")
    .add_property("number_of_outputs", &LUTTrainer::numberOfOutputs, "The dimensionality of the output vector (1 for the uni-variate case)")
    .add_property("selection_type", &LUTTrainer::selectionType, "The style for selecting features (valid for multi-variate case only)")
  ;

  // bind jesorsky loss function
  class_<JesorskyLoss>("JesorskyLoss", "The loss function to compute the Jesorsky loss", init<>())
    .def("loss", &jesorsky_loss, (arg("self"), arg("targets"), arg("scores")), "Computes the loss between the given targets and scores")
    .def("loss_gradient", &jesorsky_loss_gradient, (arg("self"), arg("targets"), arg("scores")), "Computes the loss gradient for the given targets and scores.")
    .def("loss_sum", &jesorsky_loss_sum, (arg("self"), arg("alpha"), arg("targets"), arg("previous_scores"), arg("current_scores")), "Computes the sum of the loss for the given targets and scores")
    .def("loss_gradient_sum", &jesorsky_gradient_sum, (arg("self"), arg("alpha"), arg("targets"), arg("previous_scores"), arg("current_scores")), "Computes the sum of the gradients for the given targets and scores")
  ;

  // bind auxiliary functions
//  def("weighted_histogram", &weighted_histogram1, (arg("features"), arg("weights"), arg("histogram")), "Computes the histogram of features, using the given weight for each feature.");
//  def("weighted_histogram", &weighted_histogram2, (arg("features"), arg("weights"), arg("bin_count")), "Computes and returns the histogram of features, using the given weight for each feature.");
}
