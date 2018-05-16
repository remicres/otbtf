/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelValidate_txx
#define otbTensorflowMultisourceModelValidate_txx

#include "otbTensorflowMultisourceModelValidate.h"

namespace otb
{

template <class TInputImage>
TensorflowMultisourceModelValidate<TInputImage>
::TensorflowMultisourceModelValidate()
 {
  m_BatchSize = 100;
 }


template <class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
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

  //////////////////////////////////////////////////////////////////////////////////////////
  //                               Check the references
  //////////////////////////////////////////////////////////////////////////////////////////

  const unsigned int nbOfRefs = m_References.size();
  if (nbOfRefs == 0)
    {
    itkExceptionMacro("No reference is set");
    }
  if (nbOfRefs != m_OutputFOESizes.size())
    {
    itkExceptionMacro("There is " << nbOfRefs << " but only " <<
                      m_OutputFOESizes.size() << " field of expression sizes");
    }

  // Check reference image infos
  for (unsigned int i = 0 ;i < nbOfRefs ; i++)
    {
    const SizeType outputFOESize = m_OutputFOESizes[i];
    const RegionType refRegion = m_References[i]->GetLargestPossibleRegion();
    if (refRegion.GetSize(0) != outputFOESize[0])
      {
      itkExceptionMacro("Reference image " << i << " width is " << refRegion.GetSize(0) <<
                        " but field of expression width is " << outputFOESize[0]);
      }
    if (refRegion.GetSize(1) / outputFOESize[1] != m_NumberOfSamples)
      {
      itkExceptionMacro("Reference image " << i << " height is " << refRegion.GetSize(1) <<
                        " but field of expression width is " << outputFOESize[1] <<
                        " which is not consistent with the number of samples (" << m_NumberOfSamples << ")");
      }
    }

 }

template <class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
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

template<class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
::PushBackInputReference(const ImageType *input, SizeType fieldOfExpression)
 {
  m_References.push_back(const_cast<ImageType*>(input));
  m_OutputFOESizes.push_back(fieldOfExpression);
 }

template<class TInputImage>
const TInputImage*
TensorflowMultisourceModelValidate<TInputImage>
::GetInputReference(unsigned int index)
 {
  if (m_References.size <= index || !m_References[index])
    {
    itkExceptionMacro("There is no input reference #" << index);
    }

  return static_cast<const ImageType*>(m_References[index]);
 }

template <class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
::ClearInputReferences()
 {
  m_References.clear();
  m_OutputFOESizes.clear();
 }

/**
 * Perform the validation
 */
template <class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
::GenerateData()
 {

  // Temporary images for outputs
  m_ConfusionMatrices.clear();
  m_MapsOfClasses.clear();
  std::vector<MatMapType> confMatMaps;
  for (auto const& ref: m_References)
    {
    (void) ref;

    // New confusion matrix
    MatMapType mat;
    confMatMaps.push_back(mat);
    }

  // Batches loop
  const tensorflow::uint64 nBatches = vcl_ceil(m_NumberOfSamples / m_BatchSize);
  const tensorflow::uint64 rest = m_NumberOfSamples % m_BatchSize;
  itk::ProgressReporter progress(this, 0, nBatches);
  for (tensorflow::uint64 batch = 0 ; batch < nBatches ; batch++)
    {
    // Update progress
    this->UpdateProgress((float) batch / (float) nBatches);

    // Sample start of this batch
    const tensorflow::uint64 sampleStart = batch * m_BatchSize;
    tensorflow::uint64 batchSize = m_BatchSize;
    if (rest != 0)
    {
      batchSize = rest;
    }

    // Create input tensors list
    DictListType inputs;

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
        IndexType start;
        start[0] = 0;
        start[1] = samplePos * sz_y;
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

    // Perform the validation
    if (outputs.size() != m_References.size())
      {
      itkWarningMacro("There is " << outputs.size() << " outputs returned after session run, " <<
                      "but only " << m_References.size() << " reference(s) set");
      }

    for (unsigned int refIdx = 0 ; refIdx < outputs.size() ; refIdx++)
      {
      // Recopy the chunk
      const SizeType outputFOESize = m_OutputFOESizes[refIdx];
      IndexType cpyStart;
      cpyStart.Fill(0);
      IndexType refRegStart;
      refRegStart.Fill(0);
      refRegStart[1] = outputFOESize[1] * sampleStart;
      SizeType cpySize;
      cpySize[0] = outputFOESize[0];
      cpySize[1] = outputFOESize[1] * batchSize;
      RegionType cpyRegion(cpyStart, cpySize);
      RegionType refRegion(refRegStart, cpySize);

      // Allocate a temporary image
      ImagePointerType img = ImageType::New();
      img->SetRegions(cpyRegion);
      img->SetNumberOfComponentsPerPixel(1);
      img->Allocate();

      int co = 0;
      tf::CopyTensorToImageRegion<TInputImage>(outputs[refIdx], cpyRegion, img, cpyRegion, co);

      // Retrieve the reference image region
      tf::PropagateRequestedRegion<TInputImage>(m_References[refIdx], refRegion);

      // Update the confusion matrices
      IteratorType inIt(img, cpyRegion);
      IteratorType refIt(m_References[refIdx], refRegion);
      for (inIt.GoToBegin(), refIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++refIt)
        {
        const int classIn = static_cast<LabelValueType>(inIt.Get()[0]);
        const int classRef = static_cast<LabelValueType>(refIt.Get()[0]);

        if (confMatMaps[refIdx].count(classRef) == 0)
          {
          MapType newMap;
          newMap[classIn] = 1;
          confMatMaps[refIdx][classRef] = newMap;
          }
        else
          {
          if (confMatMaps[refIdx][classRef].count(classIn) == 0)
            {
            confMatMaps[refIdx][classRef][classIn] = 1;
            }
          else
            {
            confMatMaps[refIdx][classRef][classIn]++;
            }
          }
        }
      }
    progress.CompletedPixel();
    } // Next batch

  // Compute confusion matrices
  for (unsigned int i = 0 ; i < confMatMaps.size() ; i++)
    {
    // Confusion matrix (map) for current target
    MatMapType mat = confMatMaps[i];

    // List all values
    MapOfClassesType values;
    LabelValueType curVal = 0;
    for (auto const& ref: mat)
      {
      if (values.count(ref.first) == 0)
        {
        values[ref.first] = curVal;
        curVal++;
        }
      for (auto const& in: ref.second)
        if (values.count(in.first) == 0)
          {
          values[in.first] = curVal;
          curVal++;
          }
      }

    // Build the confusion matrix
    const LabelValueType nValues = values.size();
    ConfMatType matrix(nValues, nValues);
    matrix.Fill(0);
    for (auto const& ref: mat)
      for (auto const& in: ref.second)
        matrix[values[ref.first]][values[in.first]] = in.second;

    // Add the confusion matrix
    m_ConfusionMatrices.push_back(matrix);
    m_MapsOfClasses.push_back(values);

    }

 }

template <class TInputImage>
const typename TensorflowMultisourceModelValidate<TInputImage>::ConfMatType
TensorflowMultisourceModelValidate<TInputImage>
::GetConfusionMatrix(unsigned int target)
 {
  if (target >= m_ConfusionMatrices.size())
    {
    itkExceptionMacro("Unable to get confusion matrix #" << target << ". " <<
        "There is only " << m_ConfusionMatrices.size() << " available.");
    }

  return m_ConfusionMatrices[target];
 }

template <class TInputImage>
const typename TensorflowMultisourceModelValidate<TInputImage>::MapOfClassesType
TensorflowMultisourceModelValidate<TInputImage>
::GetMapOfClasses(unsigned int target)
 {
  if (target >= m_MapsOfClasses.size())
    {
    itkExceptionMacro("Unable to get confusion matrix #" << target << ". " <<
        "There is only " << m_MapsOfClasses.size() << " available.");
    }

  return m_MapsOfClasses[target];
 }

} // end namespace otb


#endif
