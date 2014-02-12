#include <bob/io/HDF5File.h>
#include <boost/shared_ptr.hpp>

class WeakMachine{
  public:
    WeakMachine(){}

    virtual double forward1(const blitz::Array<uint16_t, 1>& features) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual double forward1(const blitz::Array<double, 1>& features) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    virtual void forward2(const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual void forward2(const blitz::Array<double, 1>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    virtual void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual void forward3(const blitz::Array<double, 2>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    virtual void forward4(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual void forward4(const blitz::Array<double, 2>& features, blitz::Array<double,2> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    virtual blitz::Array<int32_t,1> getIndices() const = 0;

    virtual void save(bob::io::HDF5File& file) const = 0;
    virtual void load(bob::io::HDF5File& file) = 0;
};

class StumpMachine : public WeakMachine{
  public:
    StumpMachine(double threshold, double polarity, int index);
    StumpMachine(bob::io::HDF5File& file);

    virtual double forward1(const blitz::Array<uint16_t, 1>& features) const;
    virtual void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const;

    virtual double forward1(const blitz::Array<double, 1>& features) const;
    virtual void forward3(const blitz::Array<double, 2>& features, blitz::Array<double,1> predictions) const;

    virtual blitz::Array<int32_t,1> getIndices() const;

    double getThreshold() const {return m_threshold;}
    double getPolarity() const {return m_polarity;}

    virtual void save(bob::io::HDF5File& file) const;
    virtual void load(bob::io::HDF5File& file);

  private:
    // helper function to compute the prediction
    double _predict(double f) const;
    // the LUT for the multi-variate case
    double m_threshold;
    double m_polarity;
    int32_t m_index;
};


class LUTMachine : public WeakMachine{
  public:
    LUTMachine(const blitz::Array<double,2> look_up_tables, const blitz::Array<int,1> indices);
    LUTMachine(bob::io::HDF5File& file);

    virtual double forward1(const blitz::Array<uint16_t, 1>& features) const;
    virtual void forward2(const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions) const;
    virtual void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const;
    virtual void forward4(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const;

    virtual blitz::Array<int32_t,1> getIndices() const;

    virtual void save(bob::io::HDF5File& file) const;
    virtual void load(bob::io::HDF5File& file);

    const blitz::Array<double, 2> getLut() const{return m_look_up_tables;}

  private:
    // the LUT for the multi-variate case
    blitz::Array<double,2> m_look_up_tables;
    blitz::Array<int32_t,1> m_indices;
    // for speed reasons, we also keep the LUT for the uni-variate case
    blitz::Array<double,1> m_look_up_table;
    int32_t m_index;
};


inline boost::shared_ptr<WeakMachine> loadWeakMachine(bob::io::HDF5File& file){
  std::string machine_type;
  file.getAttribute(".", "MachineType", machine_type);
  if (machine_type == "LutMachine" || machine_type == "LUTMachine"){
    return boost::shared_ptr<WeakMachine>(new LUTMachine(file));
  } else if (machine_type == "StumpMachine"){
    return boost::shared_ptr<WeakMachine>(new StumpMachine(file));
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

    // predicts the output for the given features (multi-variate case)
    void forward2(const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions) const;

    // predicts the output for the given features (uni-variate case)
    void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const;

    // predicts the output for the given features (multi-variate case)
    void forward4(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const;

    // predicts the output and the labels for the given features (uni-variate case)
    void forward5(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions, blitz::Array<double,1> labels) const;

    // predicts the output and the labels for the given features (multi-variate case)
    void forward6(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions, blitz::Array<double,2> labels) const;

    blitz::Array<int32_t,1> getIndices(int start = 0, int end = -1) const;

    const blitz::Array<double,2> getWeights() const {return m_weights;}

    const std::vector<boost::shared_ptr<WeakMachine> >& getWeakMachines() const {return m_weak_machines;}

    // writes the machine to file
    void save(bob::io::HDF5File& file) const;

    // loads the machine from file
    void load(bob::io::HDF5File& file);

    int numberOfOutputs() const {return m_weights.extent(1);}

  private:
    std::vector<boost::shared_ptr<WeakMachine> > m_weak_machines;
    blitz::Array<double,2> m_weights;
    blitz::Array<double,1> _weights;

    // shortcut to avoid allocating memory for each call of 'forward'
    mutable blitz::Array<double,1> _predictions1;
    mutable blitz::Array<double,2> _predictions2;
};

inline void weighted_histogram(const blitz::Array<uint16_t,1>& features, const blitz::Array<double,1>& weights, blitz::Array<double,1>& histogram){
  assert(features.extent(0) == weights.extent(0));
  histogram = 0.;
  for (int i = features.extent(0); i--;){
    histogram((int)features(i)) += weights(i);
  }
}
