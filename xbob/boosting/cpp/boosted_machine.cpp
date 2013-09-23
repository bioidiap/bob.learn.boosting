#include "Machines.h"
#include <sstream>

BoostedMachine::BoostedMachine() :
  m_weak_machines(),
  m_weights()
{
}

BoostedMachine::BoostedMachine(bob::io::HDF5File& file) :
  m_weak_machines(),
  m_weights()
{
  load(file);
}

void BoostedMachine::add_weak_machine1(const boost::shared_ptr<WeakMachine> weak_machine, const blitz::Array<double,1> weights){
  m_weak_machines.push_back(weak_machine);
  m_weights.resizeAndPreserve(m_weak_machines.size(), weights.extent(0));
  m_weights(m_weights.extent(0)-1, blitz::Range::all()) = weights;
  _weights.reference(m_weights(blitz::Range::all(), 0));
}

void BoostedMachine::add_weak_machine2(const boost::shared_ptr<WeakMachine> weak_machine, const double weight){
  m_weak_machines.push_back(weak_machine);
  m_weights.resizeAndPreserve(m_weak_machines.size(), 1);
  m_weights(m_weights.extent(0)-1, 0) = weight;
  _weights.reference(m_weights(blitz::Range::all(), 0));
}


double BoostedMachine::forward1(const blitz::Array<uint16_t,1>& features) const{
  double sum = 0.;
  //TODO: optimize using STL
  for (int i = m_weak_machines.size(); i--;){
    sum += _weights(i) * m_weak_machines[i]->forward1(features);
  }
  return sum;
}

void BoostedMachine::forward2(const blitz::Array<uint16_t,2>& features, blitz::Array<double,1> predictions, blitz::Array<double,1> labels) const{
  // initialize the predictions since they will be overwritten
  _predictions1.resize(predictions.shape());
  predictions = 0.;
  for (int i = m_weak_machines.size(); i--;){
    // predict locally
    m_weak_machines[i]->forward2(features, _predictions1);
    predictions += _weights(i) * _predictions1;
  }
  // get the labels
  for (int i = predictions.extent(0); i--;)
    labels(i) = (predictions(i) > 0) * 2 - 1;
}

void BoostedMachine::forward3(const blitz::Array<uint16_t,2>& features, blitz::Array<double,2> predictions, blitz::Array<double,2> labels) const{
  // initialize the predictions since they will be overwritten
  _predictions2.resize(predictions.shape());
  predictions = 0.;
  for (int i = m_weak_machines.size(); i--;){
    // predict locally
    m_weak_machines[i]->forward3(features, _predictions2);
    predictions += m_weights(i) * _predictions2;
  }
  // get the labels
  labels = -1;
  for (int i = predictions.extent(0); i--;){
    labels(i, blitz::maxIndex(predictions(i, blitz::Range::all()))[0]) = 1;
  }
}

blitz::Array<uint64_t,1> BoostedMachine::getIndices() const{
  std::set<uint64_t> indices;
  for (unsigned i = 0; i < m_weak_machines.size(); ++i){
    const blitz::Array<uint64_t,1>& ind = m_weak_machines[i]->getIndices();
    indices.insert(ind.begin(), ind.end());
  }

  blitz::Array<uint64_t,1> ret(indices.size());
  std::copy(indices.begin(), indices.end(), ret.begin());
  return ret;
}

// writes the machine to file
void BoostedMachine::save(bob::io::HDF5File& file) const{
  file.setAttribute(".", "version", 2);
  file.setArray("Weights", m_weights);
  for (unsigned i = 0; i < m_weights.size(); ++i){
    std::ostringstream fns;
    fns << "WeakMachine_" << i;
    file.createGroup(fns.str());
    file.cd(fns.str());
    m_weak_machines[i]->save(file);
    file.cd("..");
  }
}

// loads the machine from file
void BoostedMachine::load(bob::io::HDF5File& file){
  m_weak_machines.clear();

  // the weights
  m_weights.reference(file.readArray<double,2>("Weights"));
  _weights.reference(m_weights(blitz::Range::all(), 0));

  // name of the first machine
  std::string machine_name("WeakMachine_0");
  while (file.hasGroup(machine_name)){
    // load weight and machine
    file.cd(machine_name);
    m_weak_machines.push_back(loadWeakMachine(file));
    file.cd("..");
    // get name of the next machine
    std::ostringstream fns;
    fns << "WeakMachine_" << m_weak_machines.size();
    machine_name = fns.str();
  }

}

