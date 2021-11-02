/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2020 INRAE


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
  Superclass::SetCoordinateTolerance(itk::NumericTraits<double>::max() );
  Superclass::SetDirectionTolerance(itk::NumericTraits<double>::max() );
}

template <class TInputImage, class TOutputImage>
tensorflow::SignatureDef
TensorflowMultisourceModelBase<TInputImage, TOutputImage>
::GetSignatureDef()
{
  auto signatures = this->GetSavedModel()->GetSignatures();
  tensorflow::SignatureDef signature_def;
  // If serving_default key exists (which is the default for TF saved model), choose it as signature
  // Else, choose the first one
  if (signatures.size() == 0)
  {
    itkExceptionMacro("There are no available signatures for this tag-set. \n" <<
                      "Please check which tag-set to use by running  "<<
                      "`saved_model_cli show --dir your_model_dir --all`");
  }

  if (signatures.contains(tensorflow::kDefaultServingSignatureDefKey))
  {
    signature_def = signatures.at(tensorflow::kDefaultServingSignatureDefKey);
  }
  else
  {
    signature_def = signatures.begin()->second;
  }
  return signature_def;
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
    debugReport << "Tensor \"" << inputs[i].first << "\": " << tf::PrintTensorInfos(inputs[i].second) << "\n";
  }

  // Show user placeholders
  debugReport << "User placeholders:\n" ;
  for (auto& dict: this->GetUserPlaceholders())
  {
    debugReport << "Tensor \"" << dict.first << "\": " << tf::PrintTensorInfos(dict.second) << "\n" << std::endl;
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

  // Inputs corresponds to the names of placeholder, as specified when calling TensorFlowModelServe application
  // Decloud example: For TF1 model, it is specified by the user as "tower_0:s2_t". For TF2 model, it must be specified by the user as "s2_t"
  // Thus, for TF2, we must transform that to "serving_default_s2_t"
  DictType inputs_new;
  for (auto& dict: inputs)
  {
    DictElementType element = {m_UserNameToLayerNameMapping[dict.first], dict.second};
    inputs_new.push_back(element);
  }

  StringList m_OutputTensors_new;
  for (auto& name: m_OutputTensors)
  {
    m_OutputTensors_new.push_back(m_UserNameToLayerNameMapping[name]);
  }

  // Run the session, evaluating our output tensors from the graph
  auto status = this->GetSavedModel()->session.get()->Run(inputs_new, m_OutputTensors_new, m_TargetNodesNames, &outputs);
 

  if (!status.ok())
  {

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
  // Set all subelement of the model
  auto signaturedef = this->GetSignatureDef();
  for (auto& output: signaturedef.outputs())
  { 
    std::string userName = output.first.substr(0, output.first.find(":"));
    std::string layerName = output.second.name();
    m_UserNameToLayerNameMapping[userName] = layerName;
  }
  for (auto& input: signaturedef.inputs())
  { 
    std::string userName = input.first.substr(0, input.first.find(":"));
    std::string layerName = input.second.name();
    m_UserNameToLayerNameMapping[userName] = layerName;
  }

  // Get input and output tensors datatypes and shapes
  tf::GetTensorAttributes(signaturedef.inputs(), m_InputPlaceholders, m_InputTensorsShapes, m_InputTensorsDataTypes);
  tf::GetTensorAttributes(signaturedef.outputs(), m_OutputTensors, m_OutputTensorsShapes, m_OutputTensorsDataTypes);
}


} // end namespace otb


#endif
