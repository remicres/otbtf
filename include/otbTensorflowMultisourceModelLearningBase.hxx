/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2020 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelLearningBase_txx
#define otbTensorflowMultisourceModelLearningBase_txx

#include "otbTensorflowMultisourceModelLearningBase.h"

namespace otb
{

template <class TInputImage>
TensorflowMultisourceModelLearningBase<TInputImage>
::TensorflowMultisourceModelLearningBase(): m_BatchSize(100),
m_UseStreaming(false), m_NumberOfSamples(0)
 {
 }


template <class TInputImage>
void
TensorflowMultisourceModelLearningBase<TInputImage>
::GenerateOutputInformation()
 {
  Superclass::GenerateOutputInformation();

  // Set an empty output buffered region
  ImageType * outputPtr = this->GetOutput();
  RegionType nullRegion;
  nullRegion.GetModifiableSize().Fill(1);
  outputPtr->SetNumberOfComponentsPerPixel(1);
  outputPtr->SetLargestPossibleRegion( nullRegion );

  // Count the number of samples
  m_NumberOfSamples = 0;
  for (unsigned int i = 0 ; i < this->GetNumberOfInputs() ; i++)
    {
    // Input image pointer
    ImagePointerType inputPtr = const_cast<ImageType*>(this->GetInput(i));

    // Make sure input is available
    if ( inputPtr.IsNull() )
      {
      itkExceptionMacro(<< "Input " << i << " is null!");
      }

    // Update input information
    inputPtr->UpdateOutputInformation();

    // Patch size of tensor #i
    const SizeType inputPatchSize = this->GetInputReceptiveFields().at(i);

    // Input image requested region
    const RegionType reqRegion = inputPtr->GetLargestPossibleRegion();

    // Check size X
    if (inputPatchSize[0] != reqRegion.GetSize(0))
      itkExceptionMacro("Patch size for input " << i
          << " is " << inputPatchSize
          << " but input patches image size is " << reqRegion.GetSize());

    // Check size Y
    if (reqRegion.GetSize(1) % inputPatchSize[1] != 0)
      itkExceptionMacro("Input patches image must have a number of rows which is "
          << "a multiple of the patch size Y! Patches image has " << reqRegion.GetSize(1)
          << " rows but patch size Y is " <<  inputPatchSize[1] << " for input " << i);

    // Get the batch size
    const IndexValueType currNumberOfSamples = reqRegion.GetSize(1) / inputPatchSize[1];

    // Check the consistency with other inputs
    if (m_NumberOfSamples == 0)
      {
      m_NumberOfSamples = currNumberOfSamples;
      }
    else if (m_NumberOfSamples != currNumberOfSamples)
      {
      itkGenericExceptionMacro("Batch size of input " << (i-1)
          << " was " << m_NumberOfSamples
          << " but input " << i
          << " has a batch size of " << currNumberOfSamples );
      }
    } // next input
 }

template <class TInputImage>
void
TensorflowMultisourceModelLearningBase<TInputImage>
::GenerateInputRequestedRegion()
 {
  Superclass::GenerateInputRequestedRegion();

  // For each image, set the requested region
  RegionType nullRegion;
  for(unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
    {
    ImageType * inputImage = static_cast<ImageType * >( Superclass::ProcessObject::GetInput(i) );

    // If the streaming is enabled, we don't read the full image
    if (m_UseStreaming)
      {
      inputImage->SetRequestedRegion(nullRegion);
      }
    else
      {
      inputImage->SetRequestedRegion(inputImage->GetLargestPossibleRegion());
      }
    } // next image
 }

/**
 *
 */
template <class TInputImage>
void
TensorflowMultisourceModelLearningBase<TInputImage>
::GenerateData()
 {

  // Batches loop
  const IndexValueType nBatches = std::ceil(m_NumberOfSamples / m_BatchSize);
  const IndexValueType rest = m_NumberOfSamples % m_BatchSize;

  itk::ProgressReporter progress(this, 0, nBatches);

  for (IndexValueType batch = 0 ; batch < nBatches ; batch++)
    {

    // Feed dict
    DictType inputs;

    // Batch start and size
    const IndexValueType sampleStart = batch * m_BatchSize;
    IndexValueType batchSize = m_BatchSize;
    if (rest != 0 && batch == nBatches - 1)
    {
      batchSize = rest;
    }

    // Process the batch
    this->ProcessBatch(inputs, sampleStart, batchSize);

    progress.CompletedPixel();
    } // Next batch

 }

template <class TInputImage>
void
TensorflowMultisourceModelLearningBase<TInputImage>
::PopulateInputTensors(DictType & inputs, const IndexValueType & sampleStart,
    const IndexValueType & batchSize, const IndexListType & order)
 {
  const bool reorder = order.size();

  // Populate input tensors
  for (unsigned int i = 0 ; i < this->GetNumberOfInputs() ; i++)
    {
    // Input image pointer
    ImagePointerType inputPtr = const_cast<ImageType*>(this->GetInput(i));

    // Patch size of tensor #i
    const SizeType inputPatchSize = this->GetInputReceptiveFields().at(i);

    // Create the tensor for the batch
    const tensorflow::int64 sz_n = batchSize;
    const tensorflow::int64 sz_y = inputPatchSize[1];
    const tensorflow::int64 sz_x = inputPatchSize[0];
    const tensorflow::int64 sz_c = inputPtr->GetNumberOfComponentsPerPixel();
    const tensorflow::TensorShape inputTensorShape({sz_n, sz_y, sz_x, sz_c});
    tensorflow::Tensor inputTensor(this->GetInputTensorsDataTypes()[i], inputTensorShape);

    // Populate the tensor
    for (IndexValueType elem = 0 ; elem < batchSize ; elem++)
      {
      const tensorflow::uint64 samplePos = sampleStart + elem;
      IndexType start;
      start[0] = 0;
      if (reorder)
      {
        start[1] = order[samplePos] * sz_y;
      }
      else
      {
        start[1] = samplePos * sz_y;;
      }
      RegionType patchRegion(start, inputPatchSize);
      if (m_UseStreaming)
      {
        // If streaming is enabled, we need to explicitly propagate requested region
        tf::PropagateRequestedRegion<TInputImage>(inputPtr, patchRegion);
      }
      tf::RecopyImageRegionToTensorWithCast<TInputImage>(inputPtr, patchRegion, inputTensor, elem );
      }

    // Input #i : the tensor of patches (aka the batch)
    DictElementType input = { this->GetInputPlaceholders()[i], inputTensor };
    inputs.push_back(input);
    } // next input tensor
 }


} // end namespace otb


#endif
