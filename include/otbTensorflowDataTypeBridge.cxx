/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2020 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTensorflowDataTypeBridge.h"

namespace otb {
namespace tf {

//
// returns the datatype used by tensorflow
//
template<class Type>
tensorflow::DataType GetTensorflowDataType()
{
  if (typeid(Type) == typeid(bool))
  {
    return tensorflow::DT_BOOL;
  }
  else if (typeid(Type) == typeid(char))
  {
    return tensorflow::DT_INT8;
  }
  else if (typeid(Type) == typeid(unsigned char))
  {
    return tensorflow::DT_UINT8;
  }
  else if (typeid(Type) == typeid(unsigned short))
  {
    return tensorflow::DT_UINT16;
  }
  else if (typeid(Type) == typeid(short))
  {
    return tensorflow::DT_INT16;
  }
  else if (typeid(Type) == typeid(int))
  {
    return tensorflow::DT_INT32;
  }
  else if (typeid(Type) == typeid(unsigned int))
  {
    return tensorflow::DT_UINT32;
  }
  else if (typeid(Type) == typeid(long long int))
  {
    return tensorflow::DT_INT64;
  }
  else if (typeid(Type) == typeid(unsigned long long int))
  {
    return tensorflow::DT_UINT64;
  }
  else if (typeid(Type) == typeid(float))
  {
    return tensorflow::DT_FLOAT;
  }
  else if (typeid(Type) == typeid(double))
  {
    return tensorflow::DT_DOUBLE;
  }
  else
  {
    return tensorflow::DT_INVALID;
    //    itkGenericExceptionMacro("Unknown data type");
  }
}

//
// Return true if the tensor data type is correct
//
template<class Type>
bool HasSameDataType(const tensorflow::Tensor & tensor)
{
  return GetTensorflowDataType<Type>() == tensor.dtype();
}

//
// Return the datatype as string
//
tensorflow::string GetDataTypeAsString(tensorflow::DataType dt)
{
  return tensorflow::DataTypeString(dt);
}

} // end namespace tf
} // end namespace otb
