/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2020 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "otbTensorflowCopyUtils.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "itkMacro.h"

int compare(tensorflow::Tensor & t1, tensorflow::Tensor & t2)
{
  std::cout << "Compare " << t1.DebugString() << " and " << t2.DebugString() << std::endl;
  if (t1.dims() != t2.dims())
    {
    std::cout << "dims() differ!" << std::endl;
    return EXIT_FAILURE;
    }
  if (t1.dtype() != t2.dtype())
    {
    std::cout << "dtype() differ!" << std::endl;
    return EXIT_FAILURE;
    }
  if (t1.NumElements() != t2.NumElements())
    {
    std::cout << "NumElements() differ!" << std::endl;
    return EXIT_FAILURE;
    }
  for (unsigned int i = 0; i < t1.NumElements(); i++)
    if (t1.scalar<float>()(i) != t2.scalar<float>()(i))
      {
      std::cout << "scalar " << i << " differ!" << std::endl;
      return EXIT_FAILURE;
      }
  std::cout << "Tensors are equals :)" << std::endl;
  return EXIT_SUCCESS;
}

int floatValueToTensorTest(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{
  tensorflow::Tensor float_tensor = otb::tf::ValueToTensor("0.1234");
  tensorflow::Tensor float_tensor_ref(tensorflow::DT_BOOL, tensorflow::TensorShape({}));
  float_tensor_ref.scalar<float>()() = 0.1234;

  return compare(float_tensor, float_tensor_ref);
}
