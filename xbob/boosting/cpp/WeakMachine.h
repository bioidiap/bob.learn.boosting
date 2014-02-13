#ifndef XBOB_BOOSTING_WEAK_MACHINE_H
#define XBOB_BOOSTING_WEAK_MACHINE_H


#include <bob/io/HDF5File.h>

/**
 * This is the pure virtual base class for all weak machines.
 */
class WeakMachine{
  public:
    WeakMachine(){}

    // uni-variate forwarding of a single feature
    virtual double forward1(const blitz::Array<uint16_t, 1>& features) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual double forward1(const blitz::Array<double, 1>& features) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    // multi-variate forwarding of a single feature
    virtual void forward2(const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual void forward2(const blitz::Array<double, 1>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    // uni-variate forwarding of a set of features
    virtual void forward3(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual void forward3(const blitz::Array<double, 2>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    // multi-variate forwarding of a set of features
    virtual void forward4(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
    virtual void forward4(const blitz::Array<double, 2>& features, blitz::Array<double,2> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

    // the feature indices required by this weak machine
    virtual blitz::Array<int32_t,1> getIndices() const = 0;

    // machine IO
    virtual void save(bob::io::HDF5File& file) const = 0;
    virtual void load(bob::io::HDF5File& file) = 0;
};

#endif // XBOB_BOOSTING_WEAK_MACHINE_H
