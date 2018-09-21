/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowStreamerFilter_txx
#define otbTensorflowStreamerFilter_txx

#include "otbTensorflowStreamerFilter.h"
#include "itkImageAlgorithm.h"

namespace otb
{

template <class TInputImage, class TOutputImage>
TensorflowStreamerFilter<TInputImage, TOutputImage>
::TensorflowStreamerFilter()
 {
  m_OutputGridSize.Fill(1);
 }


template <class TInputImage, class TOutputImage>
void
TensorflowStreamerFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
 {
  // We intentionally break the pipeline
  ImageType * inputImage = static_cast<ImageType * >(  Superclass::ProcessObject::GetInput(0) );
  RegionType nullRegion;
  inputImage->SetRequestedRegion(nullRegion);
 }

/**
 * Compute the output image
 */
template <class TInputImage, class TOutputImage>
void
TensorflowStreamerFilter<TInputImage, TOutputImage>
::GenerateData()
 {
  // Output pointer and requested region
  OutputImageType * outputPtr = this->GetOutput();
  const RegionType outputReqRegion = outputPtr->GetRequestedRegion();
  outputPtr->SetRegions(outputReqRegion);
  outputPtr->Allocate();

  // Compute the aligned region
  RegionType region;
  for(unsigned int dim = 0; dim<OutputImageType::ImageDimension; ++dim)
    {
    // Get corners
    IndexValueType lower = outputReqRegion.GetIndex(dim);
    IndexValueType upper = lower + outputReqRegion.GetSize(dim);

    // Compute deltas between corners and the grid
    const IndexValueType deltaLo = lower % m_OutputGridSize[dim];
    const IndexValueType deltaUp = upper % m_OutputGridSize[dim];

    // Move corners to aligned positions
    lower -= deltaLo;
    if (deltaUp > 0)
      {
      upper += m_OutputGridSize[dim] - deltaUp;
      }

    // Update region
    region.SetIndex(dim, lower);
    region.SetSize(dim, upper - lower);

    }

  // Compute the number of subregions to process
  const unsigned int nbTilesX = region.GetSize(0) / m_OutputGridSize[0];
  const unsigned int nbTilesY = region.GetSize(1) / m_OutputGridSize[1];

  // Progress
  itk::ProgressReporter progress(this, 0, nbTilesX*nbTilesY);

  // For each tile, propagate the input region and recopy the output
  ImageType * inputImage = static_cast<ImageType * >(  Superclass::ProcessObject::GetInput(0) );
  unsigned int tx, ty;
  RegionType subRegion;
  subRegion.SetSize(m_OutputGridSize);
  for (ty = 0; ty < nbTilesY; ty++)
  {
    subRegion.SetIndex(1, ty*m_OutputGridSize[1] + region.GetIndex(1));
    for (tx = 0; tx < nbTilesX; tx++)
    {
      // Update the input subregion
      subRegion.SetIndex(0, tx*m_OutputGridSize[0] + region.GetIndex(0));
      inputImage->SetRequestedRegion(subRegion);
      inputImage->PropagateRequestedRegion();
      inputImage->UpdateOutputData();

      // Copy the subregion to output
      RegionType cpyRegion(subRegion);
      cpyRegion.Crop(outputReqRegion);
      itk::ImageAlgorithm::Copy( inputImage, outputPtr, cpyRegion, cpyRegion );

      progress.CompletedPixel();
    }
  }

 }


} // end namespace otb


#endif
