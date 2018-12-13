/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef MODULES_REMOTE_OTBTF_INCLUDE_OTBTENSORFLOWSAMPLINGUTILS_H_
#define MODULES_REMOTE_OTBTF_INCLUDE_OTBTENSORFLOWSAMPLINGUTILS_H_

#include "otbTensorflowCommon.h"
#include "vnl/vnl_vector.h"

namespace otb
{
namespace tf
{

template<class TImage>
class Distribution
{
public:
  typedef typename TImage::PixelType ValueType;
  typedef vnl_vector<float> CountsType;

  Distribution(unsigned int nClasses){
    m_NbOfClasses = nClasses;
    m_Dist = CountsType(nClasses, 0);

  }
  Distribution(unsigned int nClasses, float fillValue){
    m_NbOfClasses = nClasses;
    m_Dist = CountsType(nClasses, fillValue);

  }
  Distribution(){
    m_NbOfClasses = 2;
    m_Dist = CountsType(m_NbOfClasses, 0);
  }
  Distribution(const Distribution & other){
    m_Dist = other.Get();
    m_NbOfClasses = m_Dist.size();
  }
  ~Distribution(){}

  void Update(const typename TImage::PixelType & pixel)
  {
    m_Dist[pixel]++;
  }

  void Update(const Distribution & other)
  {
    const CountsType otherDist = other.Get();
    for (unsigned int c = 0 ; c < m_NbOfClasses ; c++)
      m_Dist[c] += otherDist[c];
  }

  CountsType Get() const
  {
    return m_Dist;
  }

  CountsType GetNormalized() const
  {
    const float invNorm = 1.0 / std::sqrt(dot_product(m_Dist, m_Dist));
    const CountsType normalizedDist = invNorm * m_Dist;
    return normalizedDist;
  }

  float Cosinus(const Distribution & other) const
  {
    return dot_product(other.GetNormalized(), GetNormalized());
  }

  std::string ToString()
  {
    std::stringstream ss;
    ss << "\n";
    for (unsigned int c = 0 ; c < m_NbOfClasses ; c++)
      ss << "\tClass #" << c << " : " << m_Dist[c] << "\n";
    return ss.str();
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

#include "otbTensorflowSamplingUtils.cxx"

#endif /* MODULES_REMOTE_OTBTF_INCLUDE_OTBTENSORFLOWSAMPLINGUTILS_H_ */
