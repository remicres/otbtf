/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelBase_txx
#define otbTensorflowMultisourceModelBase_txx

#include "otbTensorflowMultisourceModelBase.h"

namespace otb
{

template <class TInputImage, class TOutputImage>
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::TensorflowMultisourceModelBase()
 {
  m_Session = nullptr;
 }

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::PushBackInputTensorBundle(std::string placeholder, SizeType receptiveField, ImagePointerType image)
 {
  Superclass::PushBackInput(image);
  m_InputReceptiveFields.push_back(receptiveField);
  m_InputPlaceholders.push_back(placeholder);
 }

template <class TInputImage, class TOutputImage>
std::stringstream
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::GenerateDebugReport(DictType & inputs)
 {
  // Create a debug report
  std::stringstream debugReport;

  // Describe the output buffered region
  ImagePointerType outputPtr = this->GetOutput();
  const RegionType outputReqRegion = outputPtr->GetRequestedRegion();
  debugReport << "Output image buffered region: " << outputReqRegion << "\n";

  // Describe inputs
  for (unsigned int i = 0 ; i < this->GetNumberOfInputs() ; i++)
    {
    const ImagePointerType inputPtr = const_cast<TInputImage*>(this->GetInput(i));
    const RegionType reqRegion = inputPtr->GetRequestedRegion();
    debugReport << "Input #" << i << ":\n";
    debugReport << "Requested region: " << reqRegion << "\n";
    debugReport << "Tensor shape (\"" << inputs[i].first << "\": " << tf::PrintTensorShape(inputs[i].second.shape()) << "\n";
    }

  // Show user placeholders
  debugReport << "User placeholders:\n" ;
  for (auto& dict: this->GetUserPlaceholders())
    {
    debugReport << dict.first << " " << tf::PrintTensorInfos(dict.second) << "\n" << std::endl;
    }

  return debugReport;
 }


template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::RunSession(DictType & inputs, TensorListType & outputs)
 {

  // Add the user's placeholders
  for (auto& dict: this->GetUserPlaceholders())
    {
    inputs.push_back(dict);
    }

  // Run the TF session here
  // The session will initialize the outputs

  // Run the session, evaluating our output tensors from the graph
  auto status = this->GetSession()->Run(inputs, m_OutputTensors, m_TargetNodesNames, &outputs);
  if (!status.ok()) {

    // Create a debug report
    std::stringstream debugReport = GenerateDebugReport(inputs);

    // Throw an exception with the report
    itkExceptionMacro("Can't run the tensorflow session !\n" <<
                      "Tensorflow error message:\n" << status.ToString() << "\n"
                      "OTB Filter debug message:\n" << debugReport.str() );

  }

 }

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::GenerateOutputInformation()
 {

  // Check that the number of the following is the same
  // - input placeholders names
  // - input receptive fields
  // - input images
  const unsigned int nbInputs = this->GetNumberOfInputs();
  if (nbInputs != m_InputReceptiveFields.size() || nbInputs != m_InputPlaceholders.size())
    {
    itkExceptionMacro("Number of input images is " << nbInputs <<
                      " but the number of input patches size is " << m_InputReceptiveFields.size() <<
                      " and the number of input tensors names is " << m_InputPlaceholders.size());
    }

  // Check that the number of the following is the same
  // - output tensors names
  // - output expression fields
  if (m_OutputExpressionFields.size() != m_OutputTensors.size())
    {
    itkExceptionMacro("Number of output tensors names is " << m_OutputTensors.size() <<
                      " but the number of output fields of expression is " << m_OutputExpressionFields.size());
    }

  //////////////////////////////////////////////////////////////////////////////////////////
  //                               Get tensors information
  //////////////////////////////////////////////////////////////////////////////////////////

  // Get input and output tensors datatypes and shapes
  tf::GetTensorAttributes(m_Graph, m_InputPlaceholders, m_InputTensorsShapes, m_InputTensorsDataTypes);
  tf::GetTensorAttributes(m_Graph, m_OutputTensors, m_OutputTensorsShapes, m_OutputTensorsDataTypes);

 }


} // end namespace otb


#endif
