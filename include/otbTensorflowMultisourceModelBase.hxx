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
 }

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::PushBackInputBundle(std::string placeholder, SizeType fieldOfView, ImagePointerType image)
 {
  Superclass::PushBackInput(image);
  m_InputFOVSizes.push_back(fieldOfView);
  m_InputPlaceholdersNames.push_back(placeholder);
 }

template <class TInputImage, class TOutputImage>
std::stringstream
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::GenerateDebugReport(DictListType & inputs, TensorListType & outputs)
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
::RunSession(DictListType & inputs, TensorListType & outputs)
 {

  // Add the user's placeholders
  for (auto& dict: this->GetUserPlaceholders())
    {
    inputs.push_back(dict);
    }

  // Run the TF session here
  // The session will initialize the outputs

  // Run the session, evaluating our output tensors from the graph
  auto status = this->GetSession()->Run(inputs, m_OutputTensorsNames, m_TargetNodesNames, &outputs);
  if (!status.ok()) {

    // Create a debug report
    std::stringstream debugReport = GenerateDebugReport(inputs, outputs);

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
  // - placeholders names
  // - patches sizes
  // - input image
  const unsigned int nbInputs = this->GetNumberOfInputs();
  if (nbInputs != m_InputFOVSizes.size() || nbInputs != m_InputPlaceholdersNames.size())
    {
    itkExceptionMacro("Number of input images is " << nbInputs <<
                      " but the number of input patches size is " << m_InputFOVSizes.size() <<
                      " and the number of input tensors names is " << m_InputPlaceholdersNames.size());
    }

  //////////////////////////////////////////////////////////////////////////////////////////
  //                               Get tensors information
  //////////////////////////////////////////////////////////////////////////////////////////

  // Get input and output tensors datatypes and shapes
  tf::GetTensorAttributes(m_Graph, m_InputPlaceholdersNames, m_InputTensorsShapes, m_InputTensorsDataTypes);
  tf::GetTensorAttributes(m_Graph, m_OutputTensorsNames, m_OutputTensorsShapes, m_OutputTensorsDataTypes);

 }


} // end namespace otb


#endif
