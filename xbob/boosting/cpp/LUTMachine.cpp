#include "Bindings.h"
#include "LUTMachine.h"

#include <bob/core/cast.h>
#include <assert.h>
#include <set>

LUTMachine::LUTMachine(const blitz::Array<double,1> look_up_table, const int index):
  m_look_up_tables(look_up_table.extent(0), 1),
  m_indices(1,1),
  _look_up_table(),
  _index(index)
{
  // we have to copy the array, otherwise weird things happen
  m_look_up_tables(0, blitz::Range::all()) = look_up_table;
  m_indices(0) = index;
  // for the shortcut, we just reference the first row of the the look up tables
  _look_up_table.reference(m_look_up_tables(blitz::Range::all(),0));
  _index = m_indices(0);
}

LUTMachine::LUTMachine(const blitz::Array<double,2> look_up_tables, const blitz::Array<int,1> indices):
  m_look_up_tables(look_up_tables.shape()),
  m_indices(indices.shape()),
  _look_up_table(),
  _index(0)
{
  // we have to copy the array, otherwise weird things happen
  m_look_up_tables = look_up_tables;
  m_indices = indices;
  // for the shortcut, we just reference the first row of the the look up tables
  _look_up_table.reference(m_look_up_tables(blitz::Range::all(),0));
  _index = m_indices(0);
}

LUTMachine::LUTMachine(bob::io::HDF5File& file):
  m_look_up_tables(),
  m_indices(),
  _look_up_table(),
  _index(0)
{
  load(file);
}

double LUTMachine::forward(const blitz::Array<uint16_t,1>& features) const{
  // univariate, single feature
  assert ( features.extent(0) > m_index );
  assert ( features((int)m_index) < _look_up_table.extent(0) );
  return _look_up_table((int)features(_index));
}


void LUTMachine::forward(const blitz::Array<uint16_t,1>& features, blitz::Array<double,1> predictions) const{
  // multi-variate, single feature
  assert ( m_indices.extent(0) == predictions.extent(0) );
  for (int j = 0; j < m_indices.extent(0); ++j){
    assert ( features.extent(0) > m_indices(j) );
  }
  for (int j = 0; j < m_indices.extent(0); ++j){
    predictions(j) = m_look_up_tables((int)features(m_indices(j)), j);
  }
}

void LUTMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,1> predictions) const{
  // univariate, several features
  assert ( predictions.extent(0) == features.extent(0) );
  assert ( features.extent(1) > _index );
  for (int i = features.extent(0); i--;)
    assert ( features(i, (int)_index) < _look_up_table.extent(0) );
  for (int i = 0; i < features.extent(0); ++i){
    predictions(i) = _look_up_table((int)features(i, _index));
  }
}

void LUTMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,2> predictions) const{
  // multi-variate, several features
  assert ( predictions.extent(0) == features.extent(0) );
  assert ( predictions.extent(1) == m_indices.extent(0) );
  assert ( m_look_up_tables.extent(1) == m_indices.extent(0) );
  for (int j = m_indices.extent(0); j--;){
    assert ( features.extent(1) > m_indices(j) );
  }

  for (int i = 0; i < features.extent(0); ++i){
    for (int j = 0; j < m_indices.extent(0); ++j){
      predictions(i,j) = m_look_up_tables((int)features(i, m_indices(j)), j);
    }
  }
}

blitz::Array<int32_t,1> LUTMachine::getIndices() const{
  std::set<int32_t> indices;
  for (int i = 0; i < m_indices.extent(0); ++i){
    indices.insert(m_indices(i));
  }
  blitz::Array<int32_t, 1> ret(indices.size());
  std::copy(indices.begin(), indices.end(), ret.begin());
  return ret;
}

void LUTMachine::load(bob::io::HDF5File& file){
  try{
    m_look_up_tables.reference(file.readArray<double,2>("LUT"));
  }catch (std::exception){
    m_look_up_tables.reference(bob::core::array::cast<double>(file.readArray<int32_t,2>("LUT")));
  }
  try{
    m_indices.reference(file.readArray<int,1>("Indices"));
  }catch (std::exception){
    m_indices.reference(bob::core::array::cast<int>(file.readArray<int32_t,1>("Indices")));
  }

  _look_up_table.reference(m_look_up_tables(blitz::Range::all(), 0));
  _index = m_indices(0);
}

void LUTMachine::save(bob::io::HDF5File& file) const{
  file.setArray("LUT", m_look_up_tables);
  file.setArray("Indices", m_indices);
  file.setAttribute(".", "MachineType", std::string("LUTMachine"));
}




////////////////////////
/// Python bindings  ///
////////////////////////



static auto lutMachine_doc = xbob::extension::ClassDoc(
  "LUTMachine",
  "A weak machine that bases it's decision on a Look-Up-Table",
  ".. todo:: Improve documentation."
)
.add_constructor(
  xbob::extension::FunctionDoc(
    "__init__",
    "Initializes a LUTMachine object",
    "",
    true
  )
  .add_prototype("look_up_table, index", "")
  .add_prototype("look_up_tables, indices", "")
  .add_prototype("hdf5", "")
  .add_parameter("look_up_table", "float <#entries>", "The look up table (for the univariate case)")
  .add_parameter("index", "int", "The index into the feature vector (for the univariate case)")
  .add_parameter("look_up_tables", "float <#entries,#outputs>", "The look up tables, one for each output dimension (for the multi-variate case)")
  .add_parameter("indices", "int <#outputs>", "The indices into the feature vector, one for each output dimension (for the multi-variate case)")
  .add_parameter("hdf5", "xbob.io.HDF5File", "The HDF5 file object to read the weak classifier from")
);


// Some functions
static int lutMachine_init(
  LUTMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  Py_ssize_t argument_count = (args ? PyTuple_Size(args) : 0) + (kwargs ? PyDict_Size(kwargs) : 0);

  try{
    switch (argument_count){
      case 1:{
        char*  kwlist[] = {c("hdf5"), NULL};
        PyBobIoHDF5FileObject* file = 0;
        if (
          PyArg_ParseTupleAndKeywords(args, kwargs,
              "O&", kwlist,
              PyBobIoHDF5File_Converter, &file
          )
        ){
          // construct from HDF5File
          auto _ = make_safe(file);
          self->base.reset(new LUTMachine(*file->f));
        } else {
          lutMachine_doc.print_usage();
          return -1;
        }
      } break;

      case 2:{
        char* kwlist1[] = {c("look_up_table"), c("index"), NULL};
        char* kwlist2[] = {c("look_up_tables"), c("indices"), NULL};
        // two ways to call; one with index being an int:
        PyObject* key1 = Py_BuildValue("s", kwlist1[0]), * key2 = Py_BuildValue("s", kwlist1[1]);
        auto _k1 = make_safe(key1), _k2 = make_safe(key2);
        if (
          (kwargs && (PyDict_Contains(kwargs, key1) || PyDict_Contains(kwargs, key2)) ) ||
          (args && PyLong_Check(PyTuple_GetItem(args, PyTuple_Size(args)-1)))
        ){
          PyBlitzArrayObject* p_lut = 0;
          long index;
          if (PyArg_ParseTupleAndKeywords(args, kwargs,
              "O&i", kwlist1,
              &PyBlitzArray_Converter, &p_lut,
              &index
            )
          ){
            auto _1 = make_safe(p_lut);
            const auto lut = PyBlitzArrayCxx_AsBlitz<double,1>(p_lut, kwlist1[0]);
            if (!lut){
              lutMachine_doc.print_usage();
              return -1;
            }
            self->base.reset(new LUTMachine(*lut, index));
          } else {
            lutMachine_doc.print_usage();
            return -1;
          }

        // ... and one with indices being an array
        } else {
          PyBlitzArrayObject* p_lut = 0, * p_indices;
          if (PyArg_ParseTupleAndKeywords(args, kwargs,
              "O&O&", kwlist2,
              &PyBlitzArray_Converter, &p_lut,
              &PyBlitzArray_Converter, &p_indices
            )
          ){
            auto _1 = make_safe(p_lut), _2 = make_safe(p_indices);
            const auto lut = PyBlitzArrayCxx_AsBlitz<double,2>(p_lut, kwlist2[0]);
            const auto indices = PyBlitzArrayCxx_AsBlitz<int,1>(p_indices, kwlist2[1]);
            if (!lut || !indices){
              lutMachine_doc.print_usage();
              return -1;
            }
            self->base.reset(new LUTMachine(*lut, *indices));
          } else {
            lutMachine_doc.print_usage();
            return -1;
          }
        }
      } break;

      default:
        lutMachine_doc.print_usage();
        PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 2 arguments, but you provided %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, argument_count);
        return -1;
    }
  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  // set the base class as well
  self->parent.base = self->base;

  return 0;
}

static void lutMachine_exit(
  LUTMachineObject* self
)
{
  // get list of arguments
  Py_TYPE(self)->tp_free((PyObject*)self);
}



static auto lutMachine_lut_doc = xbob::extension::VariableDoc(
  "lut",
  "float <#entries,#outputs>",
  "The look-up table associated with this object. In the uni-variate case, #outputs will be 1"
);

static PyObject* lutMachine_lut(
  LUTMachineObject* self,
  void*
)
{
  auto retval = self->base->getLut();
  return PyBlitzArrayCxx_AsConstNumpy(retval);
}


static auto lutMachine_forward_doc = xbob::extension::FunctionDoc(
  "forward",
  "Returns the prediction for the given feature vector(s)",
  ".. note:: The :py:func:`__call__` function is an alias for this function.\n\n"
  "This function can be called in four different ways:\n\n"
  "1. ``(uint16 <#inputs>)`` will compute and return the uni-variate prediction for a single feature vector.\n"
  "2. ``(uint16 <#samples,#inputs>, float <#samples>)`` will compute the uni-variate prediction for several feature vectors.\n"
  "3. ``(uint16 <#inputs>, float <#outputs>)`` will compute the multi-variate prediction for a single feature vector.\n"
  "4. ``(uint16 <#samples,#inputs>, float <#samples,#outputs>)`` will compute the multi-variate prediction for several feature vectors.\n",
  true
)
.add_prototype("features", "prediction")
.add_prototype("features, predictions")
.add_parameter("features", "uint16 <#inputs> or uint16 <#samples, #inputs>", "The feature vector(s) the prediction should be computed for.")
.add_parameter("predictions", "float <#samples> or float <#outputs> or float <#samples, #outputs>", "The predicted values -- see below.")
.add_return("prediction", "float", "The predicted value -- in case a single feature is provided and a single output is required")
;

template <int N1, int N2> void _forward(LUTMachineObject* self, PyBlitzArrayObject* features, PyBlitzArrayObject* predictions){
  const auto f = PyBlitzArrayCxx_AsBlitz<uint16_t,N1>(features);
  auto p = PyBlitzArrayCxx_AsBlitz<double,N2>(predictions);
  self->base->forward(*f, *p);
}

static PyObject* lutMachine_forward(
  LUTMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("features"), c("predictions"), NULL};

  PyBlitzArrayObject* p_features = 0,* p_predictions = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&|O&", kwlist,
          &PyBlitzArray_Converter, &p_features,
          &PyBlitzArray_Converter, &p_predictions)
  )
    return NULL;

  auto _1 = make_safe(p_features), _2 = make_xsafe(p_predictions);

  try{
    if (!p_predictions){
      // uni-variate, single feature
      const auto features = PyBlitzArrayCxx_AsBlitz<uint16_t,1>(p_features, kwlist[0]);
      if (!features){
        lutMachine_forward_doc.print_usage();
        PyErr_SetString(PyExc_TypeError, "When a single parameter is specified, only 1D arrays of type uint16 are supported.");
        return NULL;
      }
      return Py_BuildValue("d", self->base->forward(*features));
    }

    if (p_features->type_num != NPY_UINT16){
      PyErr_SetString(PyExc_TypeError, "The parameter 'features' only supports 1D or 2D arrays of type uint16");
      return NULL;
    }

    if (p_features->ndim == 2 && p_predictions->ndim == 1)
      _forward<2,1>(self, p_features, p_predictions);
    else if (p_features->ndim == 1 && p_predictions->ndim == 1)
      _forward<1,1>(self, p_features, p_predictions);
    else if (p_features->ndim == 2 && p_predictions->ndim == 2)
      _forward<2,2>(self, p_features, p_predictions);
    else{
      lutMachine_forward_doc.print_usage();
      PyErr_Format(PyExc_TypeError, "The number of dimensions of %s (%ld) and %s (%ld) are not supported", kwlist[0], p_features->ndim, kwlist[1], p_predictions->ndim);
      return NULL;
    }
    Py_RETURN_NONE;
  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return NULL;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return NULL;
  }
}


static auto lutMachine_getIndices_doc = xbob::extension::FunctionDoc(
  "feature_indices",
  "Returns the feature index that will be used in this weak machine",
  NULL,
  true
)
.add_prototype("", "indices")
.add_return("indices", "int32 <1>", "The feature index required by this machine")
;

static PyObject* lutMachine_getIndices(
  LUTMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)){
    lutMachine_getIndices_doc.print_usage();
    return NULL;
  }

  const auto retval = self->base->getIndices();
  return PyBlitzArrayCxx_AsConstNumpy(retval);
}


static auto lutMachine_load_doc = xbob::extension::FunctionDoc(
  "load",
  "Loads the LUT machine from the given HDF5 file",
  NULL,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", "xbob.io.HDF5File", "The HDF5 file to load this weak machine from.")
;

static PyObject* lutMachine_load(
  LUTMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("hdf5"), NULL};
  PyBobIoHDF5FileObject* file = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
        "O&", kwlist,
        PyBobIoHDF5File_Converter, &file
    )
  )
    return NULL;

  auto _1 = make_safe(file);
  self->base->load(*file->f);
  Py_RETURN_NONE;
}


static auto lutMachine_save_doc = xbob::extension::FunctionDoc(
  "save",
  "Saves the content of this machine to the given HDF5 file",
  NULL,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", "xbob.io.HDF5File", "The HDF5 file to save this weak machine to.")
;

static PyObject* lutMachine_save(
  LUTMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("hdf5"), NULL};
  PyBobIoHDF5FileObject* file = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
        "O&", kwlist,
        PyBobIoHDF5File_Converter, &file
    )
  ){
    lutMachine_save_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(file);
  self->base->save(*file->f);
  Py_RETURN_NONE;
}

// bind the class
static PyGetSetDef lutMachine_Getters[] = {
  {
    lutMachine_lut_doc.name(),
    (getter)lutMachine_lut,
    NULL,
    lutMachine_lut_doc.doc(),
    NULL
  },
  {NULL}
};

static PyMethodDef lutMachine_Methods[] = {
  {
    lutMachine_forward_doc.name(),
    (PyCFunction)lutMachine_forward,
    METH_VARARGS | METH_KEYWORDS,
    lutMachine_forward_doc.doc(),
  },
  {
    lutMachine_getIndices_doc.name(),
    (PyCFunction)lutMachine_getIndices,
    METH_VARARGS | METH_KEYWORDS,
    lutMachine_getIndices_doc.doc(),
  },
  {
    lutMachine_load_doc.name(),
    (PyCFunction)lutMachine_load,
    METH_VARARGS | METH_KEYWORDS,
    lutMachine_load_doc.doc(),
  },
  {
    lutMachine_save_doc.name(),
    (PyCFunction)lutMachine_save,
    METH_VARARGS | METH_KEYWORDS,
    lutMachine_save_doc.doc(),
  },
  {NULL}
};


// Define Jesorsky Loss Type object; will be filled later
PyTypeObject LUTMachineType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};


static PyObject* lutMachineCreate(boost::shared_ptr<WeakMachine> machine){
  PyObject* o = LUTMachineType.tp_alloc(&LUTMachineType,0);
  reinterpret_cast<LUTMachineObject*>(o)->base = boost::dynamic_pointer_cast<LUTMachine>(machine);
  reinterpret_cast<LUTMachineObject*>(o)->parent.base = machine;
  return o;
}


bool init_LUTMachine(PyObject* module)
{

  // initialize the JesorskyLossType struct
  LUTMachineType.tp_name = lutMachine_doc.name();
  LUTMachineType.tp_basicsize = sizeof(LUTMachineObject);
  LUTMachineType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  LUTMachineType.tp_doc = lutMachine_doc.doc();
  LUTMachineType.tp_base = &WeakMachineType;

  // set the functions
  LUTMachineType.tp_new = PyType_GenericNew;
  LUTMachineType.tp_init = reinterpret_cast<initproc>(lutMachine_init);
  LUTMachineType.tp_dealloc = reinterpret_cast<destructor>(lutMachine_exit);
  LUTMachineType.tp_call = reinterpret_cast<ternaryfunc>(lutMachine_forward);
  LUTMachineType.tp_getset = lutMachine_Getters;
  LUTMachineType.tp_methods = lutMachine_Methods;

  // register machine
  if (!registerMachineType(typeid(LUTMachine).hash_code(), &lutMachineCreate))
    return false;

  // check that everyting is fine
  if (PyType_Ready(&LUTMachineType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&LUTMachineType);
  return PyModule_AddObject(module, lutMachine_doc.name(), (PyObject*)&LUTMachineType) >= 0;
}

PyObject* lutMachineCreate(WeakMachine* machine){
  LUTMachineObject* o = PyObject_New(LUTMachineObject, &LUTMachineType);
  return reinterpret_cast<PyObject*>(o);
}


