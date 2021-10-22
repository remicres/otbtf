/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2020 INRAE


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
 }

template <class TInputImage>
void
TensorflowMultisourceModelTrain<TInputImage>
::GenerateData()
 {

  // Initial sequence 1...N
  m_RandomIndices.resize(this->GetNumberOfSamples());
  std::iota (std::begin(m_RandomIndices), std::end(m_RandomIndices), 0);

  // Shuffle the sequence
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(m_RandomIndices.begin(), m_RandomIndices.end(), g);

  // Call the generic method
  Superclass::GenerateData();

 }

template <class TInputImage>
void
TensorflowMultisourceModelTrain<TInputImage>
::ProcessBatch(DictType & inputs, const IndexValueType & sampleStart,
    const IndexValueType & batchSize)
 {
  // Populate input tensors
  this->PopulateInputTensors(inputs, sampleStart, batchSize, m_RandomIndices);

  // Run the TF session here
  TensorListType outputs;
  this->RunSession(inputs, outputs);

  // Display outputs tensors
  for (auto& o: outputs)
  {
    tf::PrintTensorInfos(o);
  }

 }


} // end namespace otb


#endif
