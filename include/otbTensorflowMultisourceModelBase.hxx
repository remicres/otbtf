/*=========================================================================

  Copyright (c) 2018-2019 Remi Cresson (IRSTEA)
  Copyright (c) 2020-2021 Remi Cresson (INRAE)


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
  Superclass::SetCoordinateTolerance(itk::NumericTraits<double>::max() );
  Superclass::SetDirectionTolerance(itk::NumericTraits<double>::max() );
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


  // DEBUG 
  for (auto& ss: m_OutputTensors)
  { std::cout << "DEBUG m_OutputTensor :" << ss << std::endl;}

  for (auto& map: m_NameToLayerNameMapping)
  {
    std::cout << "first of Mapping" << map.first << std::endl;
    std::cout << "second of Mapping" << map.second << std::endl;
  }

  // Inputs corresponds to the names of placeholder, as specified when calling TensorFlowModelServe application
  // Decloud example: For TF1 model, it is specified by the user as "tower_0:s2_t". For TF2 model, it must be specified by the user as "s2_t"
  // Thus, for TF2, we must transorm that to "serving_default_s2_t"
  DictType inputs_new;
  for (auto& dict: inputs)
  {
    DictElementType element = {m_NameToLayerNameMapping[dict.first], dict.second};
    inputs_new.push_back(element);
    std::cout << "DEBUG dans boucle name issu de inputs" << dict.first << std::endl;
    std::cout << "DEBUG m_NameToLayerNameMapping[dict.first] INPUT " << m_NameToLayerNameMapping[dict.first] << std::endl;
  }

  StringList m_OutputTensors_new;
  for (auto& name: m_OutputTensors)
  {
    std::cout << "DEBUG dans boucle name issu de m_OutputTensors " << name << std::endl;
    std::cout << "DEBUG m_NameToLayerNameMapping[name] OUTPUT " << m_NameToLayerNameMapping[name] << std::endl;
    m_OutputTensors_new.push_back(m_NameToLayerNameMapping[name]);
  }

 

  // Run the session, evaluating our output tensors from the graph
  auto status = this->GetSession()->Run(inputs_new, m_OutputTensors_new, m_TargetNodesNames, &outputs);
 
  // DEBUG
  tensorflow::Tensor& output = outputs[0];
  std::cout << "<<<<<<<<<<<<<<<<<<TENSOR INFO<<<<<<<<<<<<<<<<<<<"<<std::endl;
  std::cout << otb::tf::PrintTensorInfos(output) << std::endl;
  std::cout << "<<<<<<<<<<<<<<<<<<TENSOR INFO<<<<<<<<<<<<<<<<<<<"<<std::endl;
  std::cout << " " << std::endl;

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
  tensorflow::SignatureDef signaturedef = this->GetSignatureDef();
  for (auto& output: signaturedef.outputs())
  { 
    std::string name = output.first;
    std::string layerName = output.second.name();
    m_NameToLayerNameMapping[name] = layerName;
    std::cout << "DEBUG dans boucle output: GenerateOutputInformation pour remplir mapping " << name << std::endl;
  } 
  for (auto& input: signaturedef.inputs())
  { 
    std::string inputName = input.first;
    std::string layerName = input.second.name();
    m_NameToLayerNameMapping[inputName] = layerName;
    std::cout << "DEBUG dans boucle output: GenerateOutputInformation pour remplir mapping " << inputName << std::endl;
  }

  // Get input and output tensors datatypes and shapes
  tf::GetInputAttributes(this->GetSignatureDef(), m_InputPlaceholders, m_InputTensorsShapes, m_InputTensorsDataTypes);
  tf::GetOutputAttributes(this->GetSignatureDef(), m_OutputTensors, m_OutputTensorsShapes, m_OutputTensorsDataTypes);

 }


} // end namespace otb


#endif
