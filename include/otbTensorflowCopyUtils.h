/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWUTILS_H_
#define MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWUTILS_H_

// ITK exception
#include "itkMacro.h"

// ITK image iterators
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

// tensorflow::tensor
#include "tensorflow/core/framework/tensor.h"

// tensorflow::datatype <--> ImageType::InternalPixelType
#include "otbTensorflowDataTypeBridge.h"

// STD
#include <string>

namespace otb {
namespace tf {

// Generate a string with TensorShape infos
std::string PrintTensorShape(const tensorflow::TensorShape & shp);

// Generate a string with tensor infos
std::string PrintTensorInfos(const tensorflow::Tensor & tensor);

// Create a tensor with the good datatype
template<class TImage>
tensorflow::Tensor CreateTensor(tensorflow::TensorShape & shape);

// Populate a tensor with the buffered region of a vector image
template<class TImage>
void PopulateTensorFromBufferedVectorImage(const typename TImage::Pointer bufferedimagePtr, tensorflow::Tensor & out_tensor);

// Populate the buffered region of a vector image with a given tensor's values
template<class TImage>
void TensorToImageBuffer(const tensorflow::Tensor & tensor, typename TImage::Pointer & image);

// Recopy an VectorImage region into a 4D-shaped tensorflow::Tensor ({-1, sz_y, sz_x, sz_bands})
template<class TImage, class TValueType=typename TImage::InternalPixelType>
void RecopyImageRegionToTensor(const typename TImage::Pointer inputPtr,  const typename TImage::RegionType & region, tensorflow::Tensor & tensor, unsigned int elemIdx);

// Recopy an VectorImage region into a 4D-shaped tensorflow::Tensor (TValueType-agnostic function)
template<class TImage>
void RecopyImageRegionToTensorWithCast(const typename TImage::Pointer inputPtr,  const typename TImage::RegionType & region, tensorflow::Tensor & tensor, unsigned int elemIdx);

// Sample a centered patch
template<class TImage>
void SampleCenteredPatch(const typename TImage::Pointer inputPtr, const typename TImage::IndexType & centerIndex, const typename TImage::SizeType & patchSize, tensorflow::Tensor & tensor, unsigned int elemIdx);
template<class TImage>
void SampleCenteredPatch(const typename TImage::Pointer inputPtr, const typename TImage::PointType & centerCoord, const typename TImage::SizeType & patchSize, tensorflow::Tensor & tensor, unsigned int elemIdx);

// Return the number of channels that the output tensor will occupy in the output image
tensorflow::int64 GetNumberOfChannelsForOutputTensor(const tensorflow::Tensor & tensor);

// Copy a tensor into the image region
template<class TImage, class TValueType>
void CopyTensorToImageRegion(const tensorflow::Tensor & tensor, typename TImage::Pointer outputPtr, const typename TImage::RegionType & region, int & channelOffset);

// Copy a tensor into the image region (TValueType-agnostic version)
template<class TImage>
void CopyTensorToImageRegion(const tensorflow::Tensor & tensor, const typename TImage::RegionType & bufferRegion, typename TImage::Pointer outputPtr, const typename TImage::RegionType & outputRegion, int & channelOffset);

// Convert an expression into a dict
std::pair<std::string, tensorflow::Tensor> ExpressionToTensor(std::string expression);

} // end namespace tf
} // end namespace otb

#include "otbTensorflowCopyUtils.cxx"

#endif /* MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWUTILS_H_ */
