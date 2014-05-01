#include "Bindings.h"
#include "LUTTrainer.h"
#include "Functions.h"
#include <limits>

LUTTrainer::LUTTrainer(uint16_t maximumFeatureValue, int numberOfOutputs, SelectionStyle selectionType) :
  m_maximumFeatureValue(maximumFeatureValue),
  m_numberOfOutputs(numberOfOutputs),
  m_selectionType(selectionType),
  _luts(maximumFeatureValue, numberOfOutputs),
  _selectedIndices(numberOfOutputs),
  _gradientHistogram(maximumFeatureValue)
{
}

int32_t LUTTrainer::bestIndex(const blitz::Array<double,1>& array) const{
  double min = std::numeric_limits<double>::max();
  int32_t minIndex = -1;
  for (int i = 0; i < array.extent(0); ++i){
    if (array(i) < min){
      min = array(i);
      minIndex = i;
    }
  }
  return minIndex;
}

void LUTTrainer::weightedHistogram(const blitz::Array<uint16_t,1>& features, const blitz::Array<double,1>& weights) const{
  assert(features.extent(0) == weights.extent(0));
  _gradientHistogram = 0.;
  for (int i = features.extent(0); i--;){
    _gradientHistogram((int)features(i)) += weights(i);
  }
}

boost::shared_ptr<LUTMachine> LUTTrainer::train(const blitz::Array<uint16_t,2>& trainingFeatures, const blitz::Array<double,2>& lossGradient) const{
  int featureLength = trainingFeatures.extent(1);
  _lossSum.resize(featureLength, m_numberOfOutputs);
  // Compute the sum of the gradient based on the feature values or the loss associated with each feature index
  // Compute the loss for each feature
  for (int featureIndex = featureLength; featureIndex--;){
    for (int outputIndex = m_numberOfOutputs; outputIndex--;){
      weightedHistogram(trainingFeatures(blitz::Range::all(),featureIndex), lossGradient(blitz::Range::all(), outputIndex));
      _lossSum(featureIndex,outputIndex) = - blitz::sum(blitz::abs(_gradientHistogram));
    }
  }

  // Select the most discriminative index (or indices) for classification which minimizes the loss
  //  and compute the sum of gradient for that index
  if (m_selectionType == independent){
    // independent feature selection is used if all the dimension of output use different feature
    // each of the selected feature minimize a dimension of the loss function
    for (int outputIndex = m_numberOfOutputs; outputIndex--;){
      _selectedIndices(outputIndex) = bestIndex(_lossSum(blitz::Range::all(),outputIndex));
    }
  } else {
    // for 'shared' feature selection the loss function is summed over multiple dimensions and
    // the feature that minimized this cumulative loss is used for all the outputs
    blitz::secondIndex j;
    const blitz::Array<double,1> sum(blitz::sum(_lossSum, j));
    _selectedIndices = bestIndex(sum);
  }

  // compute the look-up-tables for the best index
  for (int outputIndex = m_numberOfOutputs; outputIndex--;){
    int selectedIndex = _selectedIndices(outputIndex);
    weightedHistogram(trainingFeatures(blitz::Range::all(), selectedIndex), lossGradient(blitz::Range::all(), outputIndex));

    for (int lutIndex = m_maximumFeatureValue; lutIndex--;){
      _luts(lutIndex, outputIndex) = (_gradientHistogram(lutIndex) > 0) * 2. - 1.;
    }
  }

  // create new weak machine
  return boost::shared_ptr<LUTMachine>(new LUTMachine(_luts.copy(), _selectedIndices.copy()));

}





////////////////////////
/// Python bindings  ///
////////////////////////



static auto lutTrainer_doc = xbob::extension::ClassDoc(
  "LUTTrainer",
  "A weak machine that bases it's decision on a Look-Up-Table",
  ".. todo:: Improve documentation."
)
.add_constructor(
  xbob::extension::FunctionDoc(
    "__init__",
    "Initializes a LUTTrainer object",
    "",
    true
  )
  .add_prototype("maximum_feature_value, [number_of_outputs, selection_style]", "")
  .add_parameter("maximum_feature_value", "int", "The number of entries in the Look-Up-Tables")
  .add_parameter("number_of_outputs", "int", "The dimensionality of the output vector; defaults to 1 for the uni-variate case")
  .add_parameter("selection_style", "str", "The way, features are selected; possible values: 'shared', 'independent'; only useful for the multi-variate case; defaults to 'independent'")
);


// Some functions
static int lutTrainer_init(
  LUTTrainerObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  try{
    char*  kwlist[] = {c("maximum_feature_value"), c("number_of_outputs"), c("selection_style"), NULL};
    uint16_t max_feat = 0;
    int num_out = 1;
    const char* style = "independent";
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
          "H|is", kwlist, &max_feat, &num_out, &style)
    ){
      lutTrainer_doc.print_usage();
      return -1;
    }

    LUTTrainer::SelectionStyle s;
    if (style == std::string("independent")) s = LUTTrainer::independent;
    else if (style == std::string("shared")) s = LUTTrainer::shared;
    else {
      lutTrainer_doc.print_usage();
      PyErr_Format(PyExc_ValueError, "The 'selection_style' parameter accepts only 'independent' or 'shared', but you used '%s'", style);
      return -1;
    }

    self->base.reset(new LUTTrainer(max_feat, num_out, s));
  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;
}

static void lutTrainer_exit(
  LUTTrainerObject* self
)
{
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}



static auto lutTrainer_labels_doc = xbob::extension::VariableDoc(
  "number_of_labels",
  "uint16",
  "The highest feature value + 1, i.e., the number of entries in the LUT"
);

static PyObject* lutTrainer_labels(
  LUTTrainerObject* self,
  void*
)
{
  return Py_BuildValue("H", self->base->maximumFeatureValue());
}

static auto lutTrainer_outputs_doc = xbob::extension::VariableDoc(
  "number_of_outputs",
  "int",
  "The dimensionality of the output vector (1 for the uni-variate case)"
);

static PyObject* lutTrainer_outputs(
  LUTTrainerObject* self,
  void*
)
{
  return Py_BuildValue("i", self->base->numberOfOutputs());
}

static auto lutTrainer_selection_doc = xbob::extension::VariableDoc(
  "selection_type",
  "str",
  "The style for selecting features (valid for multi-variate case only)"
);

static PyObject* lutTrainer_selection(
  LUTTrainerObject* self,
  void*
)
{
  switch (self->base->selectionType()) {
    case LUTTrainer::independent: return Py_BuildValue("s", "independent");
    case LUTTrainer::shared:      return Py_BuildValue("s", "shared");
  }

  // impossible
  return NULL;
}


static auto lutTrainer_train_doc = xbob::extension::FunctionDoc(
  "train",
  "Trains and returns a weak LUT machine",
  ".. todo:: Write documentation for this",
  true
)
.add_prototype("training_features, loss_gradient", "lut_machine")
.add_parameter("training_features", "uint16 <#samples, #inputs>", "The feature vectors to train the weak machine")
.add_parameter("loss_gradient", "float <#samples, #outputs>", "The gradient of the loss function for the training features")
.add_return("lut_machine", "xbob.boosting.machine.LUTMachine", "The weak machine that is obtained in the current round of boosting")
;

static PyObject* lutTrainer_train(
  LUTTrainerObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  try{
    // get list of arguments
    char* kwlist[] = {c("training_features"), c("loss_gradient"), NULL};

    PyBlitzArrayObject* p_features = 0,* p_gradient = 0;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs,
            "O&O&", kwlist,
            &PyBlitzArray_Converter, &p_features,
            &PyBlitzArray_Converter, &p_gradient)
    ){
      lutTrainer_train_doc.print_usage();
      return NULL;
    }

    auto _1 = make_safe(p_features), _2 = make_safe(p_gradient);

    auto features = PyBlitzArrayCxx_AsBlitz<uint16_t,2>(p_features, kwlist[0]);
    auto gradient = PyBlitzArrayCxx_AsBlitz<double,2>(p_gradient, kwlist[1]);

    if (!features || !gradient){
      lutTrainer_train_doc.print_usage();
      return NULL;
    }

    auto machine = self->base->train(*features, *gradient);
    return createMachine(boost::dynamic_pointer_cast<WeakMachine>(machine));

  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return NULL;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return NULL;
  }
}



// bind the class
static PyGetSetDef lutTrainer_Getters[] = {
  {
    lutTrainer_labels_doc.name(),
    (getter)lutTrainer_labels,
    NULL,
    lutTrainer_labels_doc.doc(),
    NULL
  },
  {
    lutTrainer_outputs_doc.name(),
    (getter)lutTrainer_outputs,
    NULL,
    lutTrainer_outputs_doc.doc(),
    NULL
  },
  {
    lutTrainer_selection_doc.name(),
    (getter)lutTrainer_selection,
    NULL,
    lutTrainer_selection_doc.doc(),
    NULL
  },
  {NULL}
};

static PyMethodDef lutTrainer_Methods[] = {
  {
    lutTrainer_train_doc.name(),
    (PyCFunction)lutTrainer_train,
    METH_VARARGS | METH_KEYWORDS,
    lutTrainer_train_doc.doc(),
  },
  {NULL}
};


// Define Jesorsky Loss Type object; will be filled later
PyTypeObject LUTTrainerType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};


bool init_LUTTrainer(PyObject* module)
{

  // initialize the JesorskyLossType struct
  LUTTrainerType.tp_name = lutTrainer_doc.name();
  LUTTrainerType.tp_basicsize = sizeof(LUTTrainerObject);
  LUTTrainerType.tp_flags = Py_TPFLAGS_DEFAULT;
  LUTTrainerType.tp_doc = lutTrainer_doc.doc();

  // set the functions
  LUTTrainerType.tp_new = PyType_GenericNew;
  LUTTrainerType.tp_init = reinterpret_cast<initproc>(lutTrainer_init);
  LUTTrainerType.tp_dealloc = reinterpret_cast<destructor>(lutTrainer_exit);
  LUTTrainerType.tp_getset = lutTrainer_Getters;
  LUTTrainerType.tp_methods = lutTrainer_Methods;

  // check that everyting is fine
  if (PyType_Ready(&LUTTrainerType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&LUTTrainerType);
  return PyModule_AddObject(module, lutTrainer_doc.name(), (PyObject*)&LUTTrainerType) >= 0;
}

