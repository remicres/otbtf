/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


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
 }


template <class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
::GenerateOutputInformation()
 {
  Superclass::GenerateOutputInformation();

  // Check that there is some reference
  const unsigned int nbOfRefs = m_References.size();
  if (nbOfRefs == 0)
    {
    itkExceptionMacro("No reference is set");
    }

  // Check the number of references
  SizeListType outputPatchSizes = this->GetOutputExpressionFields();
  if (nbOfRefs != outputPatchSizes.size())
    {
    itkExceptionMacro("There is " << nbOfRefs << " references but only " <<
                      outputPatchSizes.size() << " output patch sizes");
    }

  // Check reference image infos
  for (unsigned int i = 0 ; i < nbOfRefs ; i++)
    {
    const SizeType outputPatchSize = outputPatchSizes[i];
    const RegionType refRegion = m_References[i]->GetLargestPossibleRegion();
    if (refRegion.GetSize(0) != outputPatchSize[0])
      {
      itkExceptionMacro("Reference image " << i << " width is " << refRegion.GetSize(0) <<
                        " but patch size (x) is " << outputPatchSize[0]);
      }
    if (refRegion.GetSize(1) != this->GetNumberOfSamples() * outputPatchSize[1])
      {
      itkExceptionMacro("Reference image " << i << " height is " << refRegion.GetSize(1) <<
                        " but patch size (y) is " << outputPatchSize[1] <<
                        " which is not consistent with the number of samples (" << this->GetNumberOfSamples() << ")");
      }
    }

 }


/*
 * Set the references images
 */
template<class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
::SetInputReferences(ImageListType input)
 {
  m_References = input;
 }

/*
 * Retrieve the i-th reference image
 * An exception is thrown if it doesn't exist.
 */
template<class TInputImage>
typename TensorflowMultisourceModelValidate<TInputImage>::ImagePointerType
TensorflowMultisourceModelValidate<TInputImage>
::GetInputReference(unsigned int index)
 {
  if (m_References.size <= index || !m_References[index])
    {
    itkExceptionMacro("There is no input reference #" << index);
    }

  return m_References[index];
 }

/**
 * Perform the validation
 * The session is ran over the entire set of batches.
 * Output is then validated against the references images,
 * and a confusion matrix is built.
 */
template <class TInputImage>
void
TensorflowMultisourceModelValidate<TInputImage>
::GenerateData()
 {

  // Temporary images for outputs
  m_ConfusionMatrices.clear();
  m_MapsOfClasses.clear();
  m_ConfMatMaps.clear();
  for (auto const& ref: m_References)
    {
    (void) ref;

    // New confusion matrix
    MatMapType mat;
    m_ConfMatMaps.push_back(mat);
    }

  // Run all the batches
  Superclass::GenerateData();

  // Compute confusion matrices
  for (unsigned int i = 0 ; i < m_ConfMatMaps.size() ; i++)
    {
    // Confusion matrix (map) for current target
    MatMapType mat = m_ConfMatMaps[i];

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
void
TensorflowMultisourceModelValidate<TInputImage>
::ProcessBatch(DictType & inputs, const IndexValueType & sampleStart,
    const IndexValueType & batchSize)
 {
  // Populate input tensors
  IndexListType empty;
  this->PopulateInputTensors(inputs, sampleStart, batchSize, empty);

  // Run the TF session here
  TensorListType outputs;
  this->RunSession(inputs, outputs);

  // Perform the validation
  if (outputs.size() != m_References.size())
    {
    itkWarningMacro("There is " << outputs.size() << " outputs returned after session run, " <<
                    "but only " << m_References.size() << " reference(s) set");
    }
  SizeListType outputEFSizes = this->GetOutputExpressionFields();
  for (unsigned int refIdx = 0 ; refIdx < outputs.size() ; refIdx++)
    {
    // Recopy the chunk
    const SizeType outputFOESize = outputEFSizes[refIdx];
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

      if (m_ConfMatMaps[refIdx].count(classRef) == 0)
        {
        MapType newMap;
        newMap[classIn] = 1;
        m_ConfMatMaps[refIdx][classRef] = newMap;
        }
      else
        {
        if (m_ConfMatMaps[refIdx][classRef].count(classIn) == 0)
          {
          m_ConfMatMaps[refIdx][classRef][classIn] = 1;
          }
        else
          {
          m_ConfMatMaps[refIdx][classRef][classIn]++;
          }
        }
      }
    }

 }

/*
 * Get the confusion matrix
 * If the target is not in the map, an exception is thrown.
 */
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

/*
 * Get the map of classes
 * If the target is not in the map, an exception is thrown.
 */
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
