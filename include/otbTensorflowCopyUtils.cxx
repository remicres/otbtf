/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTensorflowCopyUtils.h"

namespace otb
{
namespace tf
{

//
// Display a TensorShape
//
std::string
PrintTensorShape(const tensorflow::TensorShape & shp)
{
  std::stringstream s;
  unsigned int      nDims = shp.dims();
  s << "{" << shp.dim_size(0);
  for (unsigned int d = 1; d < nDims; d++)
    s << ", " << shp.dim_size(d);
  s << "}";
  return s.str();
}

//
// Display infos about a tensor
//
std::string
PrintTensorInfos(const tensorflow::Tensor & tensor)
{
  std::stringstream s;
  s << "Tensor ";
  // Show dims
  s << "shape is " << PrintTensorShape(tensor.shape());
  // Data type
  s << " data type is " << tensor.dtype();
  s << " (" << tf::GetDataTypeAsString(tensor.dtype()) << ")";
  return s.str();
}

//
// Create a tensor with the good datatype
//
template <class TImage>
tensorflow::Tensor
CreateTensor(tensorflow::TensorShape & shape)
{
  tensorflow::DataType ts_dt = GetTensorflowDataType<typename TImage::InternalPixelType>();
  tensorflow::Tensor   out_tensor(ts_dt, shape);

  return out_tensor;
}

//
// Populate a tensor with the buffered region of a vector image using std::copy
// Warning: tensor datatype must be consistent with the image value type
//
template <class TImage>
void
PopulateTensorFromBufferedVectorImage(const typename TImage::Pointer bufferedimagePtr, tensorflow::Tensor & out_tensor)
{
  size_t n_elem =
    bufferedimagePtr->GetNumberOfComponentsPerPixel() * bufferedimagePtr->GetBufferedRegion().GetNumberOfPixels();
  std::copy_n(
    bufferedimagePtr->GetBufferPointer(), n_elem, out_tensor.flat<typename TImage::InternalPixelType>().data());
}

//
// Recopy an VectorImage region into a 4D-shaped tensorflow::Tensor ({-1, sz_y, sz_x, sz_bands})
//
template <class TImage, class TValueType = typename TImage::InternalPixelType>
void
RecopyImageRegionToTensor(const typename TImage::Pointer      inputPtr,
                          const typename TImage::RegionType & region,
                          tensorflow::Tensor &                tensor,
                          unsigned int                        elemIdx) // element position along the 1st dimension
{
  typename itk::ImageRegionConstIterator<TImage> inIt(inputPtr, region);
  unsigned int                                   nBands = inputPtr->GetNumberOfComponentsPerPixel();
  auto                                           tMap = tensor.tensor<TValueType, 4>();
  for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt)
  {
    const int y = inIt.GetIndex()[1] - region.GetIndex()[1];
    const int x = inIt.GetIndex()[0] - region.GetIndex()[0];

    for (unsigned int band = 0; band < nBands; band++)
      tMap(elemIdx, y, x, band) = inIt.Get()[band];
  }
}

//
// Type-agnostic version of the 'RecopyImageRegionToTensor' function
// TODO: add some numeric types
//
template <class TImage>
void
RecopyImageRegionToTensorWithCast(const typename TImage::Pointer      inputPtr,
                                  const typename TImage::RegionType & region,
                                  tensorflow::Tensor &                tensor,
                                  unsigned int elemIdx) // element position along the 1st dimension
{
  tensorflow::DataType dt = tensor.dtype();
  if (dt == tensorflow::DT_FLOAT)
    RecopyImageRegionToTensor<TImage, float>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_DOUBLE)
    RecopyImageRegionToTensor<TImage, double>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_UINT64)
    RecopyImageRegionToTensor<TImage, unsigned long long int>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_INT64)
    RecopyImageRegionToTensor<TImage, long long int>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_UINT32)
    RecopyImageRegionToTensor<TImage, unsigned int>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_INT32)
    RecopyImageRegionToTensor<TImage, int>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_UINT16)
    RecopyImageRegionToTensor<TImage, unsigned short int>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_INT16)
    RecopyImageRegionToTensor<TImage, short int>(inputPtr, region, tensor, elemIdx);
  else if (dt == tensorflow::DT_UINT8)
    RecopyImageRegionToTensor<TImage, unsigned char>(inputPtr, region, tensor, elemIdx);
  else
    itkGenericExceptionMacro("TF DataType " << dt << " not currently implemented !");
}

//
// Sample a centered patch (from index)
//
template <class TImage>
void
SampleCenteredPatch(const typename TImage::Pointer     inputPtr,
                    const typename TImage::IndexType & centerIndex,
                    const typename TImage::SizeType &  patchSize,
                    tensorflow::Tensor &               tensor,
                    unsigned int                       elemIdx)
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
template <class TImage>
void
SampleCenteredPatch(const typename TImage::Pointer     inputPtr,
                    const typename TImage::PointType & centerCoord,
                    const typename TImage::SizeType &  patchSize,
                    tensorflow::Tensor &               tensor,
                    unsigned int                       elemIdx)
{
  // Assuming tensor is of shape {-1, sz_y, sz_x, sz_bands}
  // Get the index of the center
  typename TImage::IndexType centerIndex;
  inputPtr->TransformPhysicalPointToIndex(centerCoord, centerIndex);
  SampleCenteredPatch<TImage>(inputPtr, centerIndex, patchSize, tensor, elemIdx);
}

//
// Return the number of channels from the TensorShapeProto
// shape {n}          --> 1 (e.g. a label)
// shape {n, c}       --> c (e.g. a pixel)
// shape {n, x, y}    --> 1 (e.g. a mono-channel patch)
// shape {n, x, y, c} --> c (e.g. a multi-channel patch)
//
tensorflow::int64
GetNumberOfChannelsFromShapeProto(const tensorflow::TensorShapeProto & proto)
{
  const int nDims = proto.dim_size();
  if (nDims == 1)
    // e.g. a batch prediction, as flat tensor
    return 1;
  if (nDims == 3)
    // typically when the last dimension in squeezed following a
    // computation that does not keep dimensions (e.g. reduce_sum, etc.)
    return 1;
  // any other dimension: we assume that the last dimension represent the
  // number of channels in the output image.
  tensorflow::int64 nbChannels = proto.dim(nDims - 1).size();
  if (nbChannels < 1)
    itkGenericExceptionMacro("Cannot determine the size of the last dimension of one output tensor. Dimension index is "
                             << (nDims - 1)
                             << ". Please rewrite your model with output tensors having a shape where the last "
                                "dimension is a constant value.");
  return nbChannels;
}

//
// Copy a tensor into the image region
//
template <class TImage, class TValueType>
void
CopyTensorToImageRegion(const tensorflow::Tensor &          tensor,
                        const typename TImage::RegionType & bufferRegion,
                        typename TImage::Pointer            outputPtr,
                        const typename TImage::RegionType & outputRegion,
                        int &                               channelOffset)
{

  // Flatten the tensor
  auto tFlat = tensor.flat<TValueType>();

  // Get the number of component of the output image
  tensorflow::TensorShapeProto proto;
  tensor.shape().AsProto(&proto);
  const tensorflow::int64 outputDimSize_C = GetNumberOfChannelsFromShapeProto(proto);

  // Number of columns (size x of the buffer)
  const tensorflow::int64 nCols = bufferRegion.GetSize(0);

  // Check the tensor size vs the outputRegion size
  const tensorflow::int64 nElmT = tensor.NumElements();
  const tensorflow::int64 nElmI = bufferRegion.GetNumberOfPixels() * outputDimSize_C;
  if (nElmI != nElmT)
  {
    itkGenericExceptionMacro("Number of elements in the tensor is "
                             << nElmT << " but image outputRegion has " << nElmI << " values to fill.\n"
                             << "Buffer region is: \n"
                             << bufferRegion << "\n"
                             << "Number of components in the output image: " << outputDimSize_C << "\n"
                             << "Tensor shape: " << PrintTensorShape(tensor.shape()) << "\n"
                             << "Please check the input(s) field of view (FOV), "
                             << "the output field of expression (FOE), and the  "
                             << "output spacing scale if you run the model in fully "
                             << "convolutional mode (how many strides in your model?)");
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
    for (unsigned int c = 0; c < outputDimSize_C; c++)
      outIt.Get()[channelOffset + c] = tFlat(pos + c);
  }

  // Update the offset
  channelOffset += outputDimSize_C;
}

//
// Type-agnostic version of the 'CopyTensorToImageRegion' function
//
template <class TImage>
void
CopyTensorToImageRegion(const tensorflow::Tensor &          tensor,
                        const typename TImage::RegionType & bufferRegion,
                        typename TImage::Pointer            outputPtr,
                        const typename TImage::RegionType & region,
                        int &                               channelOffset)
{
  tensorflow::DataType dt = tensor.dtype();
  if (dt == tensorflow::DT_FLOAT)
    CopyTensorToImageRegion<TImage, float>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_DOUBLE)
    CopyTensorToImageRegion<TImage, double>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_UINT64)
    CopyTensorToImageRegion<TImage, unsigned long long int>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_INT64)
    CopyTensorToImageRegion<TImage, long long int>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_UINT32)
    CopyTensorToImageRegion<TImage, unsigned int>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_INT32)
    CopyTensorToImageRegion<TImage, int>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_UINT16)
    CopyTensorToImageRegion<TImage, unsigned short int>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_INT16)
    CopyTensorToImageRegion<TImage, short int>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else if (dt == tensorflow::DT_UINT8)
    CopyTensorToImageRegion<TImage, unsigned char>(tensor, bufferRegion, outputPtr, region, channelOffset);
  else
    itkGenericExceptionMacro("TF DataType " << dt << " not currently implemented !");
}

//
// Compare two string lowercase
//
bool
iequals(const std::string & a, const std::string & b)
{
  return std::equal(
    a.begin(), a.end(), b.begin(), b.end(), [](char cha, char chb) { return tolower(cha) == tolower(chb); });
}

// Convert a value into a tensor
// Following types are supported:
// -bool
// -int
// -float
// -vector of float
//
// e.g. "true", "0.2", "14", "(1.2, 4.2, 4)"
//
// TODO: we could add some other types (e.g. string)
tensorflow::Tensor
ValueToTensor(std::string value)
{

  std::vector<std::string> values;

  // Check if value is a vector or a scalar
  const bool has_left = (value[0] == '(');
  const bool has_right = value[value.size() - 1] == ')';

  // Check consistency
  bool is_vec = false;
  if (has_left || has_right)
  {
    is_vec = true;
    if (!has_left || !has_right)
      itkGenericExceptionMacro("Error parsing vector expression (missing parentheses ?)" << value);
  }

  // Scalar --> Vector for generic processing
  if (!is_vec)
  {
    values.push_back(value);
  }
  else
  {
    // Remove "(" and ")" chars
    std::string trimmed_value = value.substr(1, value.size() - 2);

    // Split string into vector using "," delimiter
    std::regex                 rgx("\\s*,\\s*");
    std::sregex_token_iterator iter{ trimmed_value.begin(), trimmed_value.end(), rgx, -1 };
    std::sregex_token_iterator end;
    values = std::vector<std::string>({ iter, end });
  }

  // Find type
  bool has_dot = false;
  bool is_digit = true;
  for (auto & val : values)
  {
    has_dot = has_dot || val.find(".") != std::string::npos;
    is_digit = is_digit && val.find_first_not_of("-0123456789.") == std::string::npos;
  }

  // Create tensor
  tensorflow::TensorShape shape({ values.size() });
  tensorflow::Tensor      out(tensorflow::DT_BOOL, shape);
  if (is_digit)
  {
    if (has_dot)
      out = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
    else
      out = tensorflow::Tensor(tensorflow::DT_INT32, shape);
  }

  // Fill tensor
  unsigned int idx = 0;
  for (auto & val : values)
  {

    if (is_digit)
    {
      if (has_dot)
      {
        // FLOAT
        try
        {
          out.flat<float>()(idx) = std::stof(val);
        }
        catch (...)
        {
          itkGenericExceptionMacro("Error parsing value \"" << val << "\" as float");
        }
      }
      else
      {
        // INT
        try
        {
          out.flat<int>()(idx) = std::stoi(val);
        }
        catch (...)
        {
          itkGenericExceptionMacro("Error parsing value \"" << val << "\" as int");
        }
      }
    }
    else
    {
      // BOOL
      bool ret = true;
      if (iequals(val, "true"))
      {
        ret = true;
      }
      else if (iequals(val, "false"))
      {
        ret = false;
      }
      else
      {
        itkGenericExceptionMacro("Error parsing value \"" << val << "\" as bool");
      }
      out.flat<bool>()(idx) = ret;
    }
    idx++;
  }
  otbLogMacro(Debug, << "Returning tensor: " << out.DebugString());

  return out;
}

// Convert an expression into a dict
//
// Following types are supported:
// -bool
// -int
// -float
// -vector of float
//
// e.g. is_training=true, droptout=0.2, nfeat=14, x=(1.2, 4.2, 4)
std::pair<std::string, tensorflow::Tensor>
ExpressionToTensor(std::string expression)
{
  std::pair<std::string, tensorflow::Tensor> dict;


  std::size_t found = expression.find("=");
  if (found != std::string::npos)
  {
    // Find name and value
    std::string name = expression.substr(0, found);
    std::string value = expression.substr(found + 1);

    dict.first = name;

    // Transform value into tensorflow::Tensor
    dict.second = ValueToTensor(value);
  }
  else
  {
    itkGenericExceptionMacro("The following expression is not valid: "
                             << "\n\t" << expression << ".\nExpression must be in one of the following form:"
                             << "\n- int32_value=1 \n- float_value=1.0 \n- bool_value=true."
                             << "\n- float_vec=(1.0, 5.253, 2)");
  }

  return dict;
}

} // end namespace tf
} // end namespace otb
