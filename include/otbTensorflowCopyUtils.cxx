/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTensorflowCopyUtils.h"

namespace otb {
namespace tf {

//
// Display a TensorShape
//
std::string PrintTensorShape(const tensorflow::TensorShape & shp)
{
  std::stringstream s;
  unsigned int nDims = shp.dims();
  s << "{" << shp.dim_size(0);
  for (unsigned int d = 1 ; d < nDims ; d++)
    s << ", " << shp.dim_size(d);
  s << "}" ;
  return s.str();
}

//
// Display infos about a tensor
//
std::string PrintTensorInfos(const tensorflow::Tensor & tensor)
{
  std::stringstream s;
  s << "Tensor ";
  // Show dims
  s << "shape is " << PrintTensorShape(tensor.shape());
  // Data type
  s << " data type is " << tensor.dtype();
  return s.str();
}

//
// Create a tensor with the good datatype
//
template<class TImage>
tensorflow::Tensor CreateTensor(tensorflow::TensorShape & shape)
{
  tensorflow::DataType ts_dt = GetTensorflowDataType<typename TImage::InternalPixelType>();
  tensorflow::Tensor out_tensor(ts_dt, shape);

  return out_tensor;
}

//
// Populate a tensor with the buffered region of a vector image using std::copy
// Warning: tensor datatype must be consistent with the image value type
//
template<class TImage>
void PopulateTensorFromBufferedVectorImage(const typename TImage::Pointer bufferedimagePtr, tensorflow::Tensor & out_tensor)
{
  size_t n_elem = bufferedimagePtr->GetNumberOfComponentsPerPixel() *
      bufferedimagePtr->GetBufferedRegion().GetNumberOfPixels();
  std::copy_n(bufferedimagePtr->GetBufferPointer(),
      n_elem,
      out_tensor.flat<typename TImage::InternalPixelType>().data());
}

//
// Recopy an VectorImage region into a 4D-shaped tensorflow::Tensor ({-1, sz_y, sz_x, sz_bands})
//
template<class TImage, class TValueType=typename TImage::InternalPixelType>
void RecopyImageRegionToTensor(const typename TImage::Pointer inputPtr, const typename TImage::RegionType & region,
    tensorflow::Tensor & tensor, unsigned int elemIdx) // element position along the 1st dimension
{
  typename itk::ImageRegionConstIterator<TImage> inIt(inputPtr, region);
  unsigned int nBands = inputPtr->GetNumberOfComponentsPerPixel();
  auto tMap = tensor.tensor<TValueType, 4>();
  for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt)
  {
    const int y = inIt.GetIndex()[1] - region.GetIndex()[1];
    const int x = inIt.GetIndex()[0] - region.GetIndex()[0];

    for (unsigned int band = 0 ; band < nBands ; band++)
      tMap(elemIdx, y, x, band) = inIt.Get()[band];
  }
}

//
// Type-agnostic version of the 'RecopyImageRegionToTensor' function
// TODO: add some numeric types
//
template<class TImage>
void RecopyImageRegionToTensorWithCast(const typename TImage::Pointer inputPtr, const typename TImage::RegionType & region,
    tensorflow::Tensor & tensor, unsigned int elemIdx) // element position along the 1st dimension
{
  tensorflow::DataType dt = tensor.dtype();
  if (dt == tensorflow::DT_FLOAT)
    RecopyImageRegionToTensor<TImage, float>        (inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_DOUBLE)
    RecopyImageRegionToTensor<TImage, double>       (inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_INT64)
    RecopyImageRegionToTensor<TImage, long long int>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_INT32)
    RecopyImageRegionToTensor<TImage, int>          (inputPtr, region, tensor, elemIdx);
  else
    itkGenericExceptionMacro("TF DataType "<< dt << " not currently implemented !");
}

//
// Sample a centered patch (from index)
//
template<class TImage>
void SampleCenteredPatch(const typename TImage::Pointer inputPtr, const typename TImage::IndexType & centerIndex, const typename TImage::SizeType & patchSize,
    tensorflow::Tensor & tensor, unsigned int elemIdx)
{
  typename TImage::IndexType regionStart;
  regionStart[0] = centerIndex[0] - patchSize[0] / 2;
  regionStart[1] = centerIndex[1] - patchSize[1] / 2;
  typename TImage::RegionType patchRegion(regionStart, patchSize);
  RecopyImageRegionToTensorWithCast<TImage>(inputPtr, patchRegion, tensor, elemIdx);
}

//
// Sample a centered patch (from coordinates)
//
template<class TImage>
void SampleCenteredPatch(const typename TImage::Pointer inputPtr, const typename TImage::PointType & centerCoord, const typename TImage::SizeType & patchSize,
    tensorflow::Tensor & tensor, unsigned int elemIdx)
{
  // Assuming tensor is of shape {-1, sz_y, sz_x, sz_bands}
  // Get the index of the center
  typename TImage::IndexType centerIndex;
  inputPtr->TransformPhysicalPointToIndex(centerCoord, centerIndex);
  SampleCenteredPatch<TImage>(inputPtr, centerIndex, patchSize, tensor, elemIdx);
}

// Return the number of channels that the output tensor will occupy in the output image
//
// shape {n}          --> 1 (e.g. a label)
// shape {n, c}       --> c (e.g. a vector)
// shape {x, y, c}    --> c (e.g. a patch)
// shape {n, x, y, c} --> c (e.g. some patches)
//
tensorflow::int64 GetNumberOfChannelsForOutputTensor(const tensorflow::Tensor & tensor)
{
  const tensorflow::TensorShape shape = tensor.shape();
  const int nDims = shape.dims();
  if (nDims == 1)
    return 1;
  return shape.dim_size(nDims - 1);
}

//
// Copy a tensor into the image region
// TODO: Enable to change mapping from source tensor to image to make it more generic
//
// Right now, only the following output tensor shapes can be processed:
// shape {n}          --> 1 (e.g. a label)
// shape {n, c}       --> c (e.g. a vector)
// shape {x, y, c}    --> c (e.g. a multichannel image)
//
template<class TImage, class TValueType>
void CopyTensorToImageRegion(const tensorflow::Tensor & tensor, const typename TImage::RegionType & bufferRegion,
                             typename TImage::Pointer outputPtr, const typename TImage::RegionType & outputRegion, int & channelOffset)
{

  // Flatten the tensor
  auto tFlat = tensor.flat<TValueType>();

  // Get the size of the last component of the tensor (see 'GetNumberOfChannelsForOutputTensor(...)')
  const tensorflow::int64 outputDimSize_C = GetNumberOfChannelsForOutputTensor(tensor);

  // Number of columns (size x of the buffer)
  const tensorflow::int64 nCols = bufferRegion.GetSize(0);

  // Check the tensor size vs the outputRegion size
  const tensorflow::int64 nElmT = tensor.NumElements();
  const tensorflow::int64 nElmI = bufferRegion.GetNumberOfPixels() * outputDimSize_C;
  if (nElmI != nElmT)
  {
    itkGenericExceptionMacro("Number of elements in the tensor is " << nElmT <<
        " but image outputRegion has " << nElmI <<
        " values to fill.\nBuffer region:\n" << bufferRegion <<
        "Tensor shape:\n " << PrintTensorShape(tensor.shape()) <<
        "\nPlease check the input(s) field of view (FOV), " <<
        "the output field of expression (FOE), and the  " <<
        "output spacing scale if you run the model in fully " <<
        "convolutional mode (how many strides in your model?)");
  }

  // Iterate over the image
  typename itk::ImageRegionIterator<TImage> outIt(outputPtr, outputRegion);
  for (outIt.GoToBegin(); !outIt.IsAtEnd(); ++outIt)
  {
    const int x = outIt.GetIndex()[0] - bufferRegion.GetIndex(0);
    const int y = outIt.GetIndex()[1] - bufferRegion.GetIndex(1);

    // TODO: it could be useful to change the tensor-->image mapping here.
    // e.g use a lambda for "pos" calculation
    const int pos = outputDimSize_C * (y * nCols + x);
    for (unsigned int c = 0 ; c < outputDimSize_C ; c++)
      outIt.Get()[channelOffset + c] = tFlat( pos + c);
  }

  // Update the offset
  channelOffset += outputDimSize_C;

}

//
// Type-agnostic version of the 'CopyTensorToImageRegion' function
// TODO: add some numeric types
//
template<class TImage>
void CopyTensorToImageRegion(const tensorflow::Tensor & tensor, const typename TImage::RegionType & bufferRegion,
                             typename TImage::Pointer outputPtr, const typename TImage::RegionType & region, int & channelOffset)
{
  tensorflow::DataType dt = tensor.dtype();
  if (dt == tensorflow::DT_FLOAT)
    CopyTensorToImageRegion<TImage, float>        (tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_DOUBLE)
    CopyTensorToImageRegion<TImage, double>       (tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_INT64)
    CopyTensorToImageRegion<TImage, long long int>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_INT32)
    CopyTensorToImageRegion<TImage, int>          (tensor, bufferRegion, outputPtr, region, channelOffset);
  else
    itkGenericExceptionMacro("TF DataType "<< dt << " not currently implemented !");

}

//
// Compare two string lowercase
//
bool iequals(const std::string& a, const std::string& b)
{
  return std::equal(a.begin(), a.end(),
      b.begin(), b.end(),
      [](char cha, char chb) {
    return tolower(cha) == tolower(chb);
  });
}

// Convert an expression into a dict
//
// Following types are supported:
// -bool
// -int
// -float
//
// e.g. is_training=true, droptout=0.2, nfeat=14
std::pair<std::string, tensorflow::Tensor> ExpressionToTensor(std::string expression)
{
  std::pair<std::string, tensorflow::Tensor> dict;


    std::size_t found = expression.find("=");
    if (found != std::string::npos)
    {
      // Find name and value
      std::string name = expression.substr(0, found);
      std::string value = expression.substr(found+1);

      dict.first = name;

      // Find type
      std::size_t found_dot = value.find(".") != std::string::npos;
      std::size_t is_digit = value.find_first_not_of("0123456789.") == std::string::npos;
      if (is_digit)
      {
        if (found_dot)
        {
          // FLOAT
          try
          {
            float val = std::stof(value);
            tensorflow::Tensor out(tensorflow::DT_FLOAT, tensorflow::TensorShape());
            out.scalar<float>()() = val;
            dict.second = out;

          }
          catch(...)
          {
            itkGenericExceptionMacro("Error parsing name="
                << name << " with value=" << value << " as float");
          }

        }
        else
        {
          // INT
          try
          {
            int val = std::stoi(value);
            tensorflow::Tensor out(tensorflow::DT_INT32, tensorflow::TensorShape());
            out.scalar<int>()() = val;
            dict.second = out;

          }
          catch(...)
          {
            itkGenericExceptionMacro("Error parsing name="
                << name << " with value=" << value << " as int");
          }

        }
      }
      else
      {
        // BOOL
        bool val = true;
        if (iequals(value, "true"))
        {
          val = true;
        }
        else if (iequals(value, "false"))
        {
          val = false;
        }
        else
        {
          itkGenericExceptionMacro("Error parsing name="
                          << name << " with value=" << value << " as bool");
        }
        tensorflow::Tensor out(tensorflow::DT_BOOL, tensorflow::TensorShape());
        out.scalar<bool>()() = val;
        dict.second = out;
      }

    }
    else
    {
      itkGenericExceptionMacro("The following expression is not valid: "
          << "\n\t" << expression
          << ".\nExpression must be in the form int_value=1 or float_value=1.0 or bool_value=true.");
    }

    return dict;

}

} // end namespace tf
} // end namespace otb
