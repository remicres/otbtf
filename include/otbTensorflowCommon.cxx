/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTensorflowCommon.h"

namespace otb {
namespace tf {

//
// Environment variable for the number of sources in "Multisource" applications
//
const std::string ENV_VAR_NAME_NSOURCES = "OTB_TF_NSOURCES";

//
// Get the environment variable as int
//
int GetEnvironmentVariableAsInt(const std::string & variableName)
{
  int ret = -1;
  char const* tmp = getenv( variableName.c_str() );
  if ( tmp != NULL )
  {
    std::string s( tmp );
    try
    {
      ret = std::stoi(s);
    }
    catch(...)
    {
      itkGenericExceptionMacro("Error parsing variable "
          << variableName << " as integer. Value is " << s);
    }
  }

  return ret;
}

//
// This function returns the numeric content of the ENV_VAR_NAME_NSOURCES
// environment variable
//
int GetNumberOfSources()
{
  int ret = GetEnvironmentVariableAsInt(ENV_VAR_NAME_NSOURCES);
  if (ret != -1)
  {
    return ret;
  }
  return 1;
}

//
// This function copy a patch from an input image to an output image
//
template<class TImage>
void CopyPatch(typename TImage::Pointer inputImg, typename TImage::IndexType & inputPatchIndex,
    typename TImage::Pointer outputImg, typename TImage::IndexType & outputPatchIndex,
    typename TImage::SizeType patchSize)
{
  typename TImage::RegionType inputPatchRegion(inputPatchIndex, patchSize);
  typename TImage::RegionType outputPatchRegion(outputPatchIndex, patchSize);
  typename itk::ImageRegionConstIterator<TImage> inIt (inputImg, inputPatchRegion);
  typename itk::ImageRegionIterator<TImage> outIt (outputImg, outputPatchRegion);
  for (inIt.GoToBegin(), outIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++outIt)
  {
    outIt.Set(inIt.Get());
  }
}

//
// Get image infos
//
template<class TImage>
void GetImageInfo(typename TImage::Pointer image,
    unsigned int & sizex, unsigned int & sizey, unsigned int & nBands)
{
  nBands = image->GetNumberOfComponentsPerPixel();
  sizex = image->GetLargestPossibleRegion().GetSize(0);
  sizey = image->GetLargestPossibleRegion().GetSize(1);
}

//
// Propagate the requested region in the image
//
template<class TImage>
void PropagateRequestedRegion(typename TImage::Pointer image, typename TImage::RegionType & region)
{
  image->SetRequestedRegion(region);
  image->PropagateRequestedRegion();
  image->UpdateOutputData();
}

//
// Sample an input image at the specified location
//
template<class TImage>
bool SampleImage(const typename TImage::Pointer inPtr, typename TImage::Pointer outPtr,
    typename TImage::PointType point, unsigned int elemIdx,
    typename TImage::SizeType patchSize)
{
  typename TImage::IndexType index, outIndex;
  bool canTransform = inPtr->TransformPhysicalPointToIndex(point, index);
  if (canTransform)
  {
    outIndex[0] = 0;
    outIndex[1] = elemIdx * patchSize[1];

    index[0] -= patchSize[0] / 2;
    index[1] -= patchSize[1] / 2;

    typename TImage::RegionType inPatchRegion(index, patchSize);

    if (inPtr->GetLargestPossibleRegion().IsInside(inPatchRegion))
    {
      // Fill patch
      PropagateRequestedRegion<TImage>(inPtr, inPatchRegion);
      CopyPatch<TImage>(inPtr, index, outPtr, outIndex, patchSize);

      return true;
    }
  }
  return false;

}

} // end namespace tf
} // end namespace otb
