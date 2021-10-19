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

template<typename T>
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
    if (t1.scalar<T>()(i) != t2.scalar<T>()(i))
      {
      std::cout << "scalar " << i << " differ!" << std::endl;
      return EXIT_FAILURE;
      }
  // Else
  std::cout << "Tensors are equals :)" << std::endl;
  return EXIT_SUCCESS;
}

template<typename T>
int genericValueToTensorTest(tensorflow::DataType dt, std::string expr, T value)
{
  tensorflow::Tensor t = otb::tf::ValueToTensor(expr);
  tensorflow::Tensor t_ref(dt, tensorflow::TensorShape({}));
  t_ref.scalar<T>()() = value;

  return compare<T>(t, t_ref);
}

int floatValueToTensorTest(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{
  return genericValueToTensorTest<float>(tensorflow::DT_FLOAT, "0.1234", 0.1234)
      && genericValueToTensorTest<float>(tensorflow::DT_FLOAT, "-0.1234", -0.1234) ;
}

int intValueToTensorTest(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{
  return genericValueToTensorTest<int>(tensorflow::DT_INT32, "1234", 1234)
      && genericValueToTensorTest<int>(tensorflow::DT_INT32, "-1234", -1234);
}

int boolValueToTensorTest(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{
  return genericValueToTensorTest<bool>(tensorflow::DT_BOOL, "true", true)
      && genericValueToTensorTest<bool>(tensorflow::DT_BOOL, "True", true)
      && genericValueToTensorTest<bool>(tensorflow::DT_BOOL, "False", false)
      && genericValueToTensorTest<bool>(tensorflow::DT_BOOL, "false", false);
}

template<typename T>
int genericVecValueToTensorTest(tensorflow::DataType dt, std::string expr, std::vector<T> values)
{
  tensorflow::Tensor t = otb::tf::ValueToTensor(expr);
  tensorflow::Tensor t_ref(dt, tensorflow::TensorShape({}));
  unsigned int i = 0;
  for (auto& value: values)
    {
    t_ref.scalar<T>()(i) = value;
    i++;
    }

  return compare<T>(t, t_ref);
}

int floatVecValueToTensorTest(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{
  return genericVecValueToTensorTest<float>(tensorflow::DT_FLOAT, "(0.1234, -1,-20,2.56 ,3.5)", std::vector<float>({0.1234, -1, -20, 2.56 ,3.5}));
}

int intVecValueToTensorTest(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{
  return genericVecValueToTensorTest<int>(tensorflow::DT_INT32, "(1234, -1,-20,256 ,35)", std::vector<int>({1234, -1, -20, 256 ,35}));
}

int boolVecValueToTensorTest(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{
  return genericVecValueToTensorTest<bool>(tensorflow::DT_BOOL, "(true, false,True, False", std::vector<bool>({true, false, true, false}));
}


