/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


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
TensorflowMultisourceModelBase<TInputImage, TOutputImage>::TensorflowMultisourceModelBase()
{
  Superclass::SetCoordinateTolerance(itk::NumericTraits<double>::max());
  Superclass::SetDirectionTolerance(itk::NumericTraits<double>::max());

  m_SavedModel = NULL;
}

template <class TInputImage, class TOutputImage>
tensorflow::SignatureDef
TensorflowMultisourceModelBase<TInputImage, TOutputImage>::GetSignatureDef()
{
  auto                     signatures = this->GetSavedModel()->GetSignatures();
  tensorflow::SignatureDef signature_def;

  if (signatures.size() == 0)
  {
    itkExceptionMacro("There are no available signatures for this tag-set. \n"
                      << "Please check which tag-set to use by running "
                      << "`saved_model_cli show --dir your_model_dir --all`");
  }

  // If serving_default key exists (which is the default for TF saved model), choose it as signature
  // Else, choose the first one
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
TensorflowMultisourceModelBase<TInputImage, TOutputImage>::PushBackInputTensorBundle(
  std::string       placeholder,
  SizeType          receptiveField,
  ImagePointerType  image,
  bool              useNodata,
  InternalPixelType nodataValue)
{
  Superclass::PushBackInput(image);
  m_InputReceptiveFields.push_back(receptiveField);
  m_InputPlaceholders.push_back(placeholder);
  m_InputUseNodata.push_back(useNodata);
  m_InputNodataValues.push_back(nodataValue);
}

template <class TInputImage, class TOutputImage>
std::stringstream
TensorflowMultisourceModelBase<TInputImage, TOutputImage>::GenerateDebugReport(DictType & inputs)
{
  // Create a debug report
  std::stringstream debugReport;

  // Describe the output buffered region
  ImagePointerType outputPtr = this->GetOutput();
  const RegionType outputReqRegion = outputPtr->GetRequestedRegion();
  debugReport << "Output image buffered region: " << outputReqRegion << "\n";

  // Describe inputs
  for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++)
  {
    const ImagePointerType inputPtr = const_cast<TInputImage *>(this->GetInput(i));
    const RegionType       reqRegion = inputPtr->GetRequestedRegion();
    debugReport << "Input #" << i << ":\n";
    debugReport << "Requested region: " << reqRegion << "\n";
    debugReport << "Tensor \"" << inputs[i].first << "\": " << tf::PrintTensorInfos(inputs[i].second) << "\n";
  }

  // Show user placeholders
  debugReport << "User placeholders:\n";
  for (auto & dict : this->GetUserPlaceholders())
  {
    debugReport << "Tensor \"" << dict.first << "\": " << tf::PrintTensorInfos(dict.second) << "\n" << std::endl;
  }

  return debugReport;
}

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelBase<TInputImage, TOutputImage>::RunSession(DictType & inputs, TensorListType & outputs, bool & nodata)
{

  // Run the TF session here
  // The session will initialize the outputs

  // `inputs` corresponds to a mapping {name, tensor}, with the name being specified by the user when calling
  // TensorFlowModelServe we must adapt it to `inputs_new`, that corresponds to a mapping {layerName, tensor}, with the
  // layerName being from the model
  DictType inputs_new;

  // Add the user's placeholders
  std::size_t k = 0;
  for (auto & dict : this->GetUserPlaceholders())
  {
    inputs_new.emplace_back(m_InputConstants[k], dict.second);
    k++;
  }

  // Add input tensors
  // During this step we also check for nodata values
  nodata = false;
  k = 0;
  for (auto & dict : inputs)
  {
    auto inputTensor = dict.second;
    inputs_new.emplace_back(m_InputLayers[k], inputTensor);
    if (m_InputUseNodata[k] == true)
    {
      const auto nodataValue = m_InputNodataValues[k];
      const tensorflow::int64 nElmT = inputTensor.NumElements();
      tensorflow::int64 ndCount = 0;
      auto array = inputTensor.flat<InternalPixelType>();
      for (tensorflow::int64 i = 0 ; i < nElmT ; i++)
        if (array(i) == nodataValue)
          ndCount++;
      if (ndCount == nElmT)
      {
        nodata = true;
        return;
      }
    }
    k += 1;
  }

  // Run the session, evaluating our output tensors from the graph
  auto status = this->GetSavedModel()->session.get()->Run(inputs_new, m_OutputLayers, m_TargetNodesNames, &outputs);

  if (!status.ok())
  {

    // Create a debug report
    std::stringstream debugReport = GenerateDebugReport(inputs);

    // Throw an exception with the report
    itkExceptionMacro("Can't run the tensorflow session !\n"
                      << "Tensorflow error message:\n"
                      << status.ToString()
                      << "\n"
                        "OTB Filter debug message:\n"
                      << debugReport.str());
  }
}

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelBase<TInputImage, TOutputImage>::RunSession(DictType & inputs, TensorListType & outputs)
{
  bool nodata;
  this->RunSession(inputs, outputs, nodata);
}

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelBase<TInputImage, TOutputImage>::GenerateOutputInformation()
{

  // Check that the number of the following is the same
  // - input placeholders names
  // - input receptive fields
  // - input images
  const unsigned int nbInputs = this->GetNumberOfInputs();
  if (nbInputs != m_InputReceptiveFields.size() || nbInputs != m_InputPlaceholders.size())
  {
    itkExceptionMacro("Number of input images is "
                      << nbInputs << " but the number of input patches size is " << m_InputReceptiveFields.size()
                      << " and the number of input tensors names is " << m_InputPlaceholders.size());
  }

  // Check that no-data values size is consistent with the inputs
  // If no value is specified, set a vector of the same size as the inputs
  if (m_InputNodataValues.size() == 0 && m_InputUseNodata.size() == 0)
  {
    m_InputUseNodata = BoolListType(nbInputs, false);
    m_InputNodataValues = ValueListType(nbInputs, 0.0);
  }
  if (nbInputs != m_InputNodataValues.size() || nbInputs != m_InputUseNodata.size())
  {
    itkExceptionMacro("Number of input images is " << nbInputs << " but the number of no-data values is not consistent");
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  //                               Get tensors information
  //////////////////////////////////////////////////////////////////////////////////////////
  // Set all subelement of the model
  auto signaturedef = this->GetSignatureDef();

  // Given the inputs/outputs names that the user specified, get the names of the inputs/outputs contained in the model
  // and other infos (shapes, dtypes)
  // For example, for output names specified by the user m_OutputTensors = ['s2t', 's2t_pad'],
  // this will return m_OutputLayers = ['PartitionedCall:0', 'PartitionedCall:1']
  // In case the user hasn't named the output, i.e.  m_OutputTensors = [''],
  // this will return the first output m_OutputLayers = ['PartitionedCall:0']
  StringList constantsNames;
  std::transform(m_UserPlaceholders.begin(),
                 m_UserPlaceholders.end(),
                 std::back_inserter(constantsNames),
                 [](const DictElementType & p) { return p.first; });
  if (m_UserPlaceholders.size() > 0)
  {
    // Avoid the unnecessary warning when no placeholder is fed
    tf::GetTensorAttributes(signaturedef.inputs(),
                            constantsNames,
                            m_InputConstants,
                            m_InputConstantsShapes,
                            m_InputConstantsDataTypes);
  }
  tf::GetTensorAttributes(signaturedef.inputs(),
                          m_InputPlaceholders,
                          m_InputLayers,
                          m_InputTensorsShapes,
                          m_InputTensorsDataTypes,
                          constantsNames);
  tf::GetTensorAttributes(
    signaturedef.outputs(), m_OutputTensors, m_OutputLayers, m_OutputTensorsShapes, m_OutputTensorsDataTypes);
}


} // end namespace otb


#endif
