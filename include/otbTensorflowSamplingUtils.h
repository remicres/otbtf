/*
 * otbTensorflowSamplingUtils.h
 *
 *  Created on: 12 déc. 2018
 *      Author: remi
 */

#ifndef MODULES_REMOTE_OTBTF_INCLUDE_OTBTENSORFLOWSAMPLINGUTILS_H_
#define MODULES_REMOTE_OTBTF_INCLUDE_OTBTENSORFLOWSAMPLINGUTILS_H_

#include "otbTensorflowCommon.h"
#include "otbTensorflowCopyUtils.h"
#include "vnl/vnl_vector.h"

namespace otb
{
namespace tf
{

template<class TImage>
class Distribution
{
public:
  typedef typename TImage::PixelValueType ValueType;
  typedef typename vnl_vector<unsigned int> CountsType;

  Distribution(unsigned int nClasses)
{
  m_NbOfClasses = nClasses;
  m_Dist = CountsType(nClasses, 0);

}
  ~Distribution(){}

  void Update(const typename TImage::PixelType & pixel)
  {
    m_Dist[pixel]++;
  }

  CountsType Get()
  {
    return m_Dist;
  }

private:
  unsigned int m_NbOfClasses;
  CountsType m_Dist;
};

// Update the distribution of the patch located at the specified location
template<class TImage, class TDistribution>
bool UpdateDistributionFromPatch(const typename TImage::Pointer inPtr,
                              typename TImage::PointType point, typename TImage::SizeType patchSize,
                              TDistribution & dist);

} // namesapce tf
} // namespace otb

#endif /* MODULES_REMOTE_OTBTF_INCLUDE_OTBTENSORFLOWSAMPLINGUTILS_H_ */
