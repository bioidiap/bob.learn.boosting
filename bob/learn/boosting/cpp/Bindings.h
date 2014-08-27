#ifndef XBOB_BOOSTING_BINDINGS_H
#define XBOB_BOOSTING_BINDINGS_H

#include <Python.h>

#include <xbob.blitz/cppapi.h>
#include <xbob.blitz/cleanup.h>
#include <xbob.io/api.h>
#include <xbob.extension/documentation.h>

#include <boost/shared_ptr.hpp>

// helper function to convert const char* to char*
inline char* c(const char* o){return const_cast<char*>(o);}

// Loss function
#include "LossFunction.h"

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<LossFunction> base;
} LossFunctionObject;

extern PyTypeObject LossFunctionType;

bool init_LossFunction(PyObject*);

// Jesorsky loss
#include "JesorskyLoss.h"
typedef struct {
  LossFunctionObject parent;
  boost::shared_ptr<JesorskyLoss> base;
} JesorskyLossObject;

extern PyTypeObject JesorskyLossType;

bool init_JesorskyLoss(PyObject*);


// Weak machine
#include "WeakMachine.h"

typedef PyObject*(*CreateFunction)(boost::shared_ptr<WeakMachine>);
bool registerMachineType(size_t, CreateFunction);
PyObject* createMachine(boost::shared_ptr<WeakMachine>);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<WeakMachine> base;
} WeakMachineObject;

extern PyTypeObject WeakMachineType;

int weakMachineCheck(PyObject*);

int weakMachineConverter(PyObject*, WeakMachineObject**);

bool init_WeakMachine(PyObject*);

// Stump machine
#include "StumpMachine.h"
typedef struct {
  WeakMachineObject parent;
  boost::shared_ptr<StumpMachine> base;
} StumpMachineObject;

extern PyTypeObject StumpMachineType;

bool init_StumpMachine(PyObject*);

// LUT machine
#include "LUTMachine.h"
typedef struct {
  WeakMachineObject parent;
  boost::shared_ptr<LUTMachine> base;
} LUTMachineObject;

extern PyTypeObject LUTMachineType;

bool init_LUTMachine(PyObject*);

// LUT machine
#include "BoostedMachine.h"
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<BoostedMachine> base;
} BoostedMachineObject;

extern PyTypeObject BoostedMachineType;

bool init_BoostedMachine(PyObject*);


// LUT trainer
#include "LUTTrainer.h"
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<LUTTrainer> base;
} LUTTrainerObject;

extern PyTypeObject LUTTrainerType;

bool init_LUTTrainer(PyObject*);


#endif // XBOB_BOOSTING_BINDINGS_H
