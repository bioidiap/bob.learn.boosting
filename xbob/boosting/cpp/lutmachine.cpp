#include <Machines.h>
#include <bob/core/cast.h>
#include <assert.h>

LUTMachine::LUTMachine(const blitz::Array<double,2> look_up_tables, const blitz::Array<uint64_t,1> indices):
  m_look_up_tables(look_up_tables.shape()),
  m_indices(indices.shape()),
  m_look_up_table(),
  m_index(0)
{
  // we have to copy the array, otherwise weird things happen
  m_look_up_tables = look_up_tables;
  m_indices = indices;
  // for the shortcut, we just reference the first row of the the look up tables
  m_look_up_table.reference(m_look_up_tables(blitz::Range::all(),0));
  m_index = m_indices(0);
}

LUTMachine::LUTMachine(bob::io::HDF5File& file):
  m_look_up_tables(),
  m_indices(),
  m_look_up_table(),
  m_index(0)
{
  load(file);
}

double LUTMachine::forward1(const blitz::Array<uint16_t,1>& features) const{
  assert ( features.extent(0) > m_index );
  assert ( features((int)m_index) < m_look_up_table.extent(0) );
  return m_look_up_table((int)features(m_index));
}

void LUTMachine::forward2(const blitz::Array<uint16_t,2>& features, blitz::Array<double,1> predictions) const{
  assert ( predictions.extent(0) == features.extent(0) );
  assert ( features.extent(1) > m_index );
  for (int i = features.extent(0); i--;)
    assert ( features(i, (int)m_index) < m_look_up_table.extent(0) );
  for (int i = 0; i < features.extent(0); ++i){
    predictions(i) = m_look_up_table((int)features(i, (int)m_index));
  }
}

void LUTMachine::forward3(const blitz::Array<uint16_t,2>& features, blitz::Array<double,2> predictions) const{
  assert ( predictions.extent(0) == features.extent(0) );
  assert ( predictions.extent(1) == m_indices.extent(0) );
  assert ( m_look_up_tables.extent(1) == m_indices.extent(0) );
  for (int j = m_indices.extent(0); j--;){
    assert ( features.extent(1) > m_indices(j) );
  }

  for (int i = 0; i < features.extent(0); ++i){
    for (int j = 0; j < m_indices.extent(0); ++j){
      predictions(i,j) = m_look_up_tables((int)features(i, (int)m_indices(j)), j);
    }
  }
}

blitz::Array<uint64_t,1> LUTMachine::getIndices() const{
  std::set<uint64_t> indices;
  for (int i = 0; i < m_indices.extent(0); ++i){
    indices.insert(m_indices(i));
  }
  blitz::Array<uint64_t, 1> ret(indices.size());
  std::copy(indices.begin(), indices.end(), ret.begin());
  return ret;
}

void LUTMachine::load(bob::io::HDF5File& file){
  m_look_up_tables.reference(bob::core::array::cast<double>(file.readArray<int64_t,2>("LUT")));
  m_indices.reference(bob::core::array::cast<uint64_t>(file.readArray<int64_t,1>("Indices")));
  m_look_up_table.reference(m_look_up_tables(blitz::Range::all(), 0));
  m_index = m_indices(0);
}

void LUTMachine::save(bob::io::HDF5File& file) const{
  file.setArray("LUT", m_look_up_tables);
  file.setArray("Indices", m_indices);
  file.setAttribute(".", "MachineType", "LUTMachine");
}