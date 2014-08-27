#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif // NO_IMPORT_ARRAY

#include "Bindings.h"

static const char* const module_docstr = "C++ implementations for several classes and functions in the xbob.boosting module";

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  XBOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  0,
};
#endif


PyObject*
create_module(void)
{

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
# else
  PyObject* module = Py_InitModule3(XBOB_EXT_MODULE_NAME, NULL, module_docstr);
# endif

  if (!module) return NULL;

  if (!init_LossFunction(module)) return NULL;
  if (!init_JesorskyLoss(module)) return NULL;


  if (!init_WeakMachine(module)) return NULL;
  if (!init_StumpMachine(module)) return NULL;
  if (!init_LUTMachine(module)) return NULL;
  if (!init_BoostedMachine(module)) return NULL;

  if (!init_LUTTrainer(module)) return NULL;


  /* imports C-API dependencies */
  if (import_xbob_blitz() < 0) return NULL;
  if (import_xbob_io() < 0) return NULL;

  // module was initialized successfully
  return module;
}

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}

