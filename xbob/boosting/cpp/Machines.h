#include <bob/io/HDF5File.h>
#include <boost/shared_ptr.hpp>
#include <set>

class WeakMachine{
  public:
    WeakMachine(){}

    virtual double forward1(const blitz::Array<uint16_t, 1>& features) const = 0;
    virtual void forward2(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const = 0;
    virtual void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const = 0;

    virtual blitz::Array<uint64_t,1> getIndices() const = 0;

    virtual void save(bob::io::HDF5File& file) const = 0;
    virtual void load(bob::io::HDF5File& file) = 0;
};

class LUTMachine : public WeakMachine{
  public:
    LUTMachine(const blitz::Array<double,2> look_up_tables, const blitz::Array<uint64_t,1> indices);
//    LUTMachine(const blitz::Array<double,1>& look_up_table, uint64_t index);
    LUTMachine(bob::io::HDF5File& file);

    virtual double forward1(const blitz::Array<uint16_t, 1>& features) const;
    virtual void forward2(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const;
    virtual void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const;

    virtual blitz::Array<uint64_t,1> getIndices() const;

    virtual void save(bob::io::HDF5File& file) const;
    virtual void load(bob::io::HDF5File& file);

    const blitz::Array<double, 2> getLut() const{return m_look_up_tables;}

  private:
    // the LUT for the multi-variate case
    blitz::Array<double,2> m_look_up_tables;
    blitz::Array<uint64_t,1> m_indices;
    // for speed reasons, we also keep the LUT for the uni-variate case
    blitz::Array<double,1> m_look_up_table;
    uint64_t m_index;
};

inline boost::shared_ptr<WeakMachine> loadWeakMachine(bob::io::HDF5File& file){
  std::string machine_type;
  file.getAttribute(".", "MachineType", machine_type);
  if (machine_type == "LutMachine" || machine_type == "LUTMachine"){
    return boost::shared_ptr<WeakMachine>(new LUTMachine(file));
  }
  throw std::runtime_error("Weak machine type '" + machine_type + "' is not known or supported.");
}

class BoostedMachine{
  public:
    BoostedMachine();
    BoostedMachine(bob::io::HDF5File& file);

    // adds the machine
    void add_weak_machine1(const boost::shared_ptr<WeakMachine> weak_machine, const blitz::Array<double,1> weights);
    void add_weak_machine2(const boost::shared_ptr<WeakMachine> weak_machine, const double weight);

    // predicts the output for the given single feature
    double forward1(const blitz::Array<uint16_t, 1>& features) const;

    // predicts the output and the labels for the given features (uni-variate case)
    void forward2(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions, blitz::Array<double,1> labels) const;

    // predicts the output and the labels for the given features (multi-variate case)
    void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions, blitz::Array<double,2> labels) const;

    blitz::Array<uint64_t,1> getIndices() const;

    const blitz::Array<double,2> getWeights() const {return m_weights;}

    // writes the machine to file
    void save(bob::io::HDF5File& file) const;

    // loads the machine from file
    void load(bob::io::HDF5File& file);

  private:
    std::vector<boost::shared_ptr<WeakMachine> > m_weak_machines;
    blitz::Array<double,2> m_weights;
    blitz::Array<double,1> _weights;

    // shortcut to avoid allocating memory for each call of 'forward'
    mutable blitz::Array<double,1> _predictions1;
    mutable blitz::Array<double,2> _predictions2;
};
