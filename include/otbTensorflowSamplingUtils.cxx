/*=========================================================================

  Copyright (c) 2018-2019 IRSTEA
  Copyright (c) 2020-2020 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTensorflowSamplingUtils.h"

namespace otb
{
namespace tf
{

//
// Update the distribution of the patch located at the specified location
//
template<class TImage, class TDistribution>
bool UpdateDistributionFromPatch(const typename TImage::Pointer inPtr,
    typename TImage::PointType point, typename TImage::SizeType patchSize,
    TDistribution & dist)
{
  typename TImage::IndexType index;
  bool canTransform = inPtr->TransformPhysicalPointToIndex(point, index);
  if (canTransform)
  {
    index[0] -= patchSize[0] / 2;
    index[1] -= patchSize[1] / 2;

    typename TImage::RegionType inPatchRegion(index, patchSize);

    if (inPtr->GetLargestPossibleRegion().IsInside(inPatchRegion))
    {
      // Fill patch
      PropagateRequestedRegion<TImage>(inPtr, inPatchRegion);

      typename itk::ImageRegionConstIterator<TImage> inIt (inPtr, inPatchRegion);
      for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt)
      {
        dist.Update(inIt.Get());
      }
      return true;
    }
  }
  return false;

}


} // namespace tf
} // namespace otb
