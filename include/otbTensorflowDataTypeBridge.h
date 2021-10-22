/*=========================================================================

  Copyright (c) 2018-2019 IRSTEA
  Copyright (c) 2020-2020 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWDATATYPEBRIDGE_H_
#define MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWDATATYPEBRIDGE_H_

#include <typeinfo>
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor.h"

namespace otb {
namespace tf {

// returns the datatype used by tensorflow
template<class Type>
tensorflow::DataType GetTensorflowDataType();

// Return true if the tensor data type is correct
template<class Type>
bool HasSameDataType(const tensorflow::Tensor & tensor);

// Return datatype as string
tensorflow::string GetDataTypeAsString(tensorflow::DataType dt);

} // end namespace tf
} // end namespace otb

#include "otbTensorflowDataTypeBridge.cxx"

#endif /* MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWDATATYPEBRIDGE_H_ */
