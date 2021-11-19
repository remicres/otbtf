/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWCOMMON_H_
#define MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWCOMMON_H_

// STD
#include <iostream>
#include <iterator>
#include <string>
#include <algorithm>
#include <functional>
#include "itkMacro.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

namespace otb {
namespace tf {

// Environment variable for the number of sources in "Multisource" applications
extern const std::string ENV_VAR_NAME_NSOURCES;

// Get the environment variable as int
int GetEnvironmentVariableAsInt(const std::string & variableName);

// Get the value (as int) of the environment variable ENV_VAR_NAME_NSOURCES
int GetNumberOfSources();

// This function copy a patch from an input image to an output image
template<class TImage>
void CopyPatch(typename TImage::Pointer inputImg, typename TImage::IndexType & inputPatchIndex,
    typename TImage::Pointer outputImg, typename TImage::IndexType & outputPatchIndex,
    typename TImage::SizeType patchSize);

// Get image infos
template<class TImage>
void GetImageInfo(typename TImage::Pointer image,
    unsigned int & sizex, unsigned int & sizey, unsigned int & nBands);

// Propagate the requested region in the image
template<class TImage>
void PropagateRequestedRegion(typename TImage::Pointer image, typename TImage::RegionType & region);

// Sample an input image at the specified location
template<class TImage>
bool SampleImage(const typename TImage::Pointer inPtr, typename TImage::Pointer outPtr,
    typename TImage::PointType point, unsigned int elemIdx,
    typename TImage::SizeType patchSize);

} // end namespace tf
} // end namespace otb

#include "otbTensorflowCommon.cxx"

#endif /* MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWCOMMON_H_ */
