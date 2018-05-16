/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelTrain_txx
#define otbTensorflowMultisourceModelTrain_txx

#include "otbTensorflowMultisourceModelTrain.h"

namespace otb
{

template <class TInputImage>
TensorflowMultisourceModelTrain<TInputImage>
::TensorflowMultisourceModelTrain()
 {
  m_BatchSize = 100;
 }


template <class TInputImage>
void
TensorflowMultisourceModelTrain<TInputImage>
::GenerateOutputInformation()
 {
  Superclass::GenerateOutputInformation();

  ImageType * outputPtr = this->GetOutput();
  RegionType nullRegion;
  nullRegion.GetModifiableSize().Fill(1);
  outputPtr->SetNumberOfComponentsPerPixel(1);
  outputPtr->SetLargestPossibleRegion( nullRegion );

  //////////////////////////////////////////////////////////////////////////////////////////
  //                               Check the number of samples
  //////////////////////////////////////////////////////////////////////////////////////////


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
    const SizeType inputPatchSize = this->GetInputFOVSizes().at(i);

    // Input image requested region
    const RegionType reqRegion = inputPtr->GetLargestPossibleRegion();

    // Check size X
    if (inputPatchSize[0] != reqRegion.GetSize(0))
      itkExceptionMacro("Patch size for input " << i << " is " << inputPatchSize <<
                        " but input patches image size is " << reqRegion.GetSize());

    // Check size Y
    if (reqRegion.GetSize(1) % inputPatchSize[1] != 0)
      itkExceptionMacro("Input patches image must have a number of rows which is "
          "a multiple of the patch size Y! Patches image has " << reqRegion.GetSize(1) <<
          " rows but patch size Y is " <<  inputPatchSize[1] << " for input " << i);

    // Get the batch size
    const tensorflow::uint64 currNumberOfSamples = reqRegion.GetSize(1) / inputPatchSize[1];

    // Check the consistency with other inputs
    if (m_NumberOfSamples == 0)
      {
      m_NumberOfSamples = currNumberOfSamples;
      }
    else if (m_NumberOfSamples != currNumberOfSamples)
      {
      itkGenericExceptionMacro("Previous batch size is " << m_NumberOfSamples << " but input " << i
                               << " has a batch size of " << currNumberOfSamples );
      }
    } // next input
 }

template <class TInputImage>
void
TensorflowMultisourceModelTrain<TInputImage>
::GenerateInputRequestedRegion()
 {
  Superclass::GenerateInputRequestedRegion();

  // For each image, set no image region
  for(unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
    {
    RegionType nullRegion;
    ImageType * inputImage = static_cast<ImageType * >( Superclass::ProcessObject::GetInput(i) );
    inputImage->SetRequestedRegion(nullRegion);
    } // next image
 }

/**
 *
 */
template <class TInputImage>
void
TensorflowMultisourceModelTrain<TInputImage>
::GenerateData()
 {

  // Random sequence
  std::vector<int> v(m_NumberOfSamples) ;
  std::iota (std::begin(v), std::end(v), 0);

  // Shuffle
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(v.begin(), v.end(), g);

  // Batches loop
  const tensorflow::uint64 nBatches = vcl_ceil(m_NumberOfSamples / m_BatchSize);
  const tensorflow::uint64 rest = m_NumberOfSamples % m_BatchSize;
  itk::ProgressReporter progress(this, 0, nBatches);
  for (tensorflow::uint64 batch = 0 ; batch < nBatches ; batch++)
    {
    // Update progress
    this->UpdateProgress((float) batch / (float) nBatches);

    // Create input tensors list
    DictListType inputs;

    // Batch start and size
    const tensorflow::uint64 sampleStart = batch * m_BatchSize;
    tensorflow::uint64 batchSize = m_BatchSize;
    if (rest != 0)
    {
      batchSize = rest;
    }

    // Populate input tensors
    for (unsigned int i = 0 ; i < this->GetNumberOfInputs() ; i++)
      {
      // Input image pointer
      ImagePointerType inputPtr = const_cast<ImageType*>(this->GetInput(i));

      // Patch size of tensor #i
      const SizeType inputPatchSize = this->GetInputFOVSizes().at(i);

      // Create the tensor for the batch
      const tensorflow::int64 sz_n = batchSize;
      const tensorflow::int64 sz_y = inputPatchSize[1];
      const tensorflow::int64 sz_x = inputPatchSize[0];
      const tensorflow::int64 sz_c = inputPtr->GetNumberOfComponentsPerPixel();
      const tensorflow::TensorShape inputTensorShape({sz_n, sz_y, sz_x, sz_c});
      tensorflow::Tensor inputTensor(this->GetInputTensorsDataTypes()[i], inputTensorShape);

      // Populate the tensor
      for (tensorflow::uint64 elem = 0 ; elem < batchSize ; elem++)
        {
        const tensorflow::uint64 samplePos = sampleStart + elem;
        const tensorflow::uint64 randPos = v[samplePos];
        IndexType start;
        start[0] = 0;
        start[1] = randPos * sz_y;
        RegionType patchRegion(start, inputPatchSize);
        tf::PropagateRequestedRegion<TInputImage>(inputPtr, patchRegion);
        tf::RecopyImageRegionToTensorWithCast<TInputImage>(inputPtr, patchRegion, inputTensor, elem );
        }

      // Input #i : the tensor of patches (aka the batch)
      DictType input1 = { this->GetInputPlaceholdersNames()[i], inputTensor };
      inputs.push_back(input1);
      } // next input tensor

    // Run the TF session here
    TensorListType outputs;
    this->RunSession(inputs, outputs);

    // Get output tensors
    for (auto& output: outputs)
      {
      std::cout << tf::PrintTensorInfos(output) << std::endl;
      }

    progress.CompletedPixel();
    } // Next batch

 }

} // end namespace otb


#endif
