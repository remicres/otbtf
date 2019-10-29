/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowSampler_txx
#define otbTensorflowSampler_txx

#include "otbTensorflowSampler.h"

namespace otb
{

template <class TInputImage, class TVectorData>
TensorflowSampler<TInputImage, TVectorData>
::TensorflowSampler()
 {
  m_NumberOfAcceptedSamples = 0;
  m_NumberOfRejectedSamples = 0;
  m_RejectPatchesWithNodata = false;
  m_NodataValue = 0;
 }

template <class TInputImage, class TVectorData>
void
TensorflowSampler<TInputImage, TVectorData>
::PushBackInputWithPatchSize(const ImageType *input, SizeType & patchSize)
 {
  this->ProcessObject::PushBackInput(const_cast<ImageType*>(input));
  m_PatchSizes.push_back(patchSize);
 }

template <class TInputImage, class TVectorData>
const TInputImage*
TensorflowSampler<TInputImage, TVectorData>
::GetInput(unsigned int index)
 {
  if (this->GetNumberOfInputs() < 1)
  {
    itkExceptionMacro("Input not set");
  }

  return static_cast<const ImageType*>(this->ProcessObject::GetInput(index));
 }


/**
 * Resize an image given a patch size and a number of samples
 */
template <class TInputImage, class TVectorData>
void
TensorflowSampler<TInputImage, TVectorData>
::ResizeImage(ImagePointerType & image, SizeType & patchSize, unsigned int nbSamples)
 {
  // New image region
  RegionType region;
  region.SetSize(0, patchSize[0]);
  region.SetSize(1, patchSize[1] * nbSamples);

  // Resize
  ExtractROIMultiFilterPointerType resizer = ExtractROIMultiFilterType::New();
  resizer->SetInput(image);
  resizer->SetExtractionRegion(region);
  resizer->Update();

  // Assign
  image = resizer->GetOutput();
 }

/**
 * Allocate an image given a patch size and a number of samples
 */
template <class TInputImage, class TVectorData>
void
TensorflowSampler<TInputImage, TVectorData>
::AllocateImage(ImagePointerType & image, SizeType & patchSize, unsigned int nbSamples, unsigned int nbComponents)
 {
  // Image region
  RegionType region;
  region.SetSize(0, patchSize[0]);
  region.SetSize(1, patchSize[1] * nbSamples);

  // Allocate label image
  image = ImageType::New();
  image->SetNumberOfComponentsPerPixel(nbComponents);
  image->SetRegions(region);
  image->Allocate();
 }

/**
 * Do the work
 */
template <class TInputImage, class TVectorData>
void
TensorflowSampler<TInputImage, TVectorData>
::Update()
 {

  // Check number of inputs
  if (this->GetNumberOfInputs() != m_PatchSizes.size())
  {
    itkExceptionMacro("Number of inputs and patches sizes are not the same");
  }

  // Count points
  unsigned int nTotal = 0;
  unsigned int geomId = 0;
  TreeIteratorType itVector(m_InputVectorData->GetDataTree());
  itVector.GoToBegin();
  while (!itVector.IsAtEnd())
  {
    if (!itVector.Get()->IsRoot() && !itVector.Get()->IsDocument() && !itVector.Get()->IsFolder())
    {
      const DataNodePointer currentGeometry = itVector.Get();
      if (!currentGeometry->HasField(m_Field))
      {
        itkWarningMacro("Field \"" << m_Field << "\" not found in geometry #" << geomId);
      }
      else
      {
        nTotal++;
      }
      geomId++;
    }
    ++itVector;
  } // next feature

  // Check number
  if (nTotal == 0)
  {
    itkExceptionMacro("There is no geometry to sample. Geometries must be points.")
  }

  // Allocate label image
  SizeType labelPatchSize;
  labelPatchSize.Fill(1);
  AllocateImage(m_OutputLabelImage, labelPatchSize, nTotal, 1);

  // Allocate patches image
  const unsigned int nbInputs = this->GetNumberOfInputs();
  m_OutputPatchImages.clear();
  m_OutputPatchImages.reserve(nbInputs);
  for (unsigned int i = 0 ; i < nbInputs ; i++)
  {
    ImagePointerType newImage;
    AllocateImage(newImage, m_PatchSizes[i], nTotal, GetInput(i)->GetNumberOfComponentsPerPixel());
    newImage->SetSignedSpacing(this->GetInput(i)->GetSignedSpacing());
    m_OutputPatchImages.push_back(newImage);
  }

  itk::ProgressReporter progess(this, 0, nTotal);

  // Iterate on the vector data
  itVector.GoToBegin();
  unsigned long count = 0;
  unsigned long rejected = 0;
  IndexType labelIndex;
  labelIndex[0] = 0;
  PixelType labelPix;
  labelPix.SetSize(1);
  while (!itVector.IsAtEnd())
  {
    if (!itVector.Get()->IsRoot() && !itVector.Get()->IsDocument() && !itVector.Get()->IsFolder())
    {
      DataNodePointer currentGeometry = itVector.Get();
      PointType point = currentGeometry->GetPoint();

      // Get the label value
      labelPix[0] = static_cast<InternalPixelType>(currentGeometry->GetFieldAsInt(m_Field));

      bool hasBeenSampled = true;
      for (unsigned int i = 0 ; i < nbInputs ; i++)
      {
        // Get input
        ImagePointerType inputPtr = const_cast<ImageType *>(this->GetInput(i));

        // Try to sample the image
        if (!tf::SampleImage<ImageType>(inputPtr, m_OutputPatchImages[i], point, count, m_PatchSizes[i]))
        {
          // If not, reject this sample
          hasBeenSampled = false;
        }
        // Check if it contains no-data values
        if (m_RejectPatchesWithNodata && hasBeenSampled)
          {
          IndexType outIndex;
          outIndex[0] = 0;
          outIndex[1] = count * m_PatchSizes[i][1];
          RegionType region(outIndex, m_PatchSizes[i]);

          IteratorType it(m_OutputPatchImages[i], region);
          for (it.GoToBegin(); !it.IsAtEnd(); ++it)
            {
            PixelType pix = it.Get();
            for (int i; i<pix.Size(); i++)
              if (pix[i] == m_NodataValue)
              {
                std::cout << "[0]: pix[" << i << "]=" << pix[i] << std::endl;
                std::cout << "break" << std::endl;
                hasBeenSampled = false;
                std::cout << "no actually break" << std::endl;
                break;
              }
            if (!hasBeenSampled)
              {
              std::cout << "BREAKED" << std::endl;
              break;
              }
            }

          }
      } // Next input
      if (hasBeenSampled)
      {
        // Fill label
        labelIndex[1] = count;
        m_OutputLabelImage->SetPixel(labelIndex, labelPix);

        // update count
        count++;
      }
      else
      {
        std::cout << "REJECTED: " << rejected << std::endl;
        rejected++;
      }

      // Update progres
      progess.CompletedPixel();

    }

    ++itVector;
  } // next feature

  // Resize output images
  ResizeImage(m_OutputLabelImage, labelPatchSize, count);
  for (unsigned int i = 0 ; i < nbInputs ; i++)
  {
    ResizeImage(m_OutputPatchImages[i], m_PatchSizes[i], count);
  }

  // Update number of samples produced
  m_NumberOfAcceptedSamples = count;
  m_NumberOfRejectedSamples = rejected;

 }

} // end namespace otb


#endif
