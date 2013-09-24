#include "Machines.h"
#include <bob/core/cast.h>
#include <assert.h>
#include <set>

StumpMachine::StumpMachine(double threshold, double polarity, int index):
  m_threshold(threshold),
  m_polarity(polarity),
  m_index(index)
{
}

StumpMachine::StumpMachine(bob::io::HDF5File& file):
  m_threshold(0),
  m_polarity(0),
  m_index(0)
{
  load(file);
}

double StumpMachine::_predict(double f) const{
  return m_polarity * ((-2. * (f < m_threshold)) + 1.);
}

double StumpMachine::forward1(const blitz::Array<double, 1>& features) const{
  return _predict(features((int)m_index));
}

void StumpMachine::forward2(const blitz::Array<double, 2>& features, blitz::Array<double,1> predictions) const{
  for (int i = features.extent(0); i--;){
    predictions(i) = _predict(features(i, (int)m_index));
  }
}


double StumpMachine::forward1(const blitz::Array<uint16_t, 1>& features) const{
  return _predict(features((int)m_index));
}

void StumpMachine::forward2(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const{
  for (int i = features.extent(0); i--;){
    predictions(i) = _predict(features(i, (int)m_index));
  }
}


blitz::Array<int,1> StumpMachine::getIndices() const{
  blitz::Array<int, 1> ret(1);
  ret = m_index;
  return ret;
}

void StumpMachine::load(bob::io::HDF5File& file){
  m_threshold = file.read<double>("Threshold");
  m_polarity = file.read<double>("Polarity");
  m_index = file.read<int>("Index");
}

void StumpMachine::save(bob::io::HDF5File& file) const{
  file.set("Threshold", m_threshold);
  file.set("Polarity", m_polarity);
  file.set("Index", m_index);
  file.setAttribute(".", "MachineType", "StumpMachine");
}