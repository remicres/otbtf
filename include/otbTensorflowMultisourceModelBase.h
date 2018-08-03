/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelBase_h
#define otbTensorflowMultisourceModelBase_h

#include "itkProcessObject.h"
#include "itkNumericTraits.h"
#include "itkSimpleDataObjectDecorator.h"

// Tensorflow
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

// Tensorflow helpers
#include "otbTensorflowGraphOperations.h"
#include "otbTensorflowDataTypeBridge.h"
#include "otbTensorflowCopyUtils.h"
#include "otbTensorflowCommon.h"

namespace otb
{

/**
 * \class TensorflowMultisourceModelBase
 * \brief This filter is base for TensorFlow model over multiple input images.
 *
 * The filter takes N input images and feed the TensorFlow model.
 * Names of input placeholders must be specified using the
 * SetInputPlaceholdersNames method
 *
 * TODO:
 *   Replace FOV (Field Of View) --> RF (Receptive Field)
 *   Replace FEO (Field Of Expr) --> EF (Expression Field)
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT TensorflowMultisourceModelBase :
public itk::ImageToImageFilter<TInputImage, TOutputImage>
{

public:

  /** Standard class typedefs. */
  typedef TensorflowMultisourceModelBase             Self;
  typedef itk::ProcessObject                         Superclass;
  typedef itk::SmartPointer<Self>                    Pointer;
  typedef itk::SmartPointer<const Self>              ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowMultisourceModelBase, itk::ImageToImageFilter);

  /** Images typedefs */
  typedef TInputImage                                ImageType;
  typedef typename TInputImage::Pointer              ImagePointerType;
  typedef typename TInputImage::PixelType            PixelType;
  typedef typename TInputImage::InternalPixelType    InternalPixelType;
  typedef typename TInputImage::IndexType            IndexType;
  typedef typename TInputImage::IndexValueType       IndexValueType;
  typedef typename TInputImage::PointType            PointType;
  typedef typename TInputImage::SizeType             SizeType;
  typedef typename TInputImage::SizeValueType        SizeValueType;
  typedef typename TInputImage::SpacingType          SpacingType;
  typedef typename TInputImage::RegionType           RegionType;

  /** Typedefs for parameters */
  typedef std::pair<std::string, tensorflow::Tensor> DictType;
  typedef std::vector<std::string>                   StringList;
  typedef std::vector<SizeType>                      SizeListType;
  typedef std::vector<DictType>                      DictListType;
  typedef std::vector<tensorflow::DataType>          DataTypeListType;
  typedef std::vector<tensorflow::TensorShapeProto>  TensorShapeProtoList;
  typedef std::vector<tensorflow::Tensor>            TensorListType;

  /** Set and Get the Tensorflow session and graph */
  void SetGraph(tensorflow::GraphDef graph)      { m_Graph = graph;     }
  tensorflow::GraphDef GetGraph()                { return m_Graph ;     }
  void SetSession(tensorflow::Session * session) { m_Session = session; }
  tensorflow::Session * GetSession()             { return m_Session;    }

  /** Model parameters */
  void PushBackInputBundle(std::string placeholder, SizeType receptiveField, ImagePointerType image);

//  /** Input placeholders names */
//  itkSetMacro(InputPlaceholdersNames, StringList);
  itkGetMacro(InputPlaceholdersNames, StringList);
//
//  /** Receptive field */
//  itkSetMacro(InputFOVSizes, SizeListType);
  itkGetMacro(InputFOVSizes, SizeListType);

  /** Output tensors names */
  itkSetMacro(OutputTensorsNames, StringList);
  itkGetMacro(OutputTensorsNames, StringList);

  /** Expression field */
  itkSetMacro(OutputFOESizes, SizeListType);
  itkGetMacro(OutputFOESizes, SizeListType);

  /** User placeholders */
  void SetUserPlaceholders(DictListType dict) { m_UserPlaceholders = dict; }
  DictListType GetUserPlaceholders()          { return m_UserPlaceholders; }

  /** Target nodes names */
  itkSetMacro(TargetNodesNames, StringList);
  itkGetMacro(TargetNodesNames, StringList);

  /** Read only methods */
  itkGetMacro(InputTensorsDataTypes, DataTypeListType);
  itkGetMacro(OutputTensorsDataTypes, DataTypeListType);
  itkGetMacro(InputTensorsShapes, TensorShapeProtoList);
  itkGetMacro(OutputTensorsShapes, TensorShapeProtoList);

  virtual void GenerateOutputInformation();

protected:
  TensorflowMultisourceModelBase();
  virtual ~TensorflowMultisourceModelBase() {};

  virtual std::stringstream GenerateDebugReport(DictListType & inputs, TensorListType & outputs);

  virtual void RunSession(DictListType & inputs, TensorListType & outputs);

private:
  TensorflowMultisourceModelBase(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  // Tensorflow graph and session
  tensorflow::GraphDef       m_Graph;                   // The tensorflow graph
  tensorflow::Session *      m_Session;                 // The tensorflow session

  // Model parameters
  StringList                 m_InputPlaceholdersNames;  // Input placeholders names
  SizeListType               m_InputFOVSizes;           // Input tensors field of view (FOV) sizes
  SizeListType               m_OutputFOESizes;          // Output tensors field of expression (FOE) sizes
  DictListType               m_UserPlaceholders;        // User placeholders
  StringList                 m_OutputTensorsNames;      // User tensors
  StringList                 m_TargetNodesNames;        // User target tensors

  // Read-only
  DataTypeListType           m_InputTensorsDataTypes;   // Input tensors datatype
  DataTypeListType           m_OutputTensorsDataTypes;  // Output tensors datatype
  TensorShapeProtoList       m_InputTensorsShapes;      // Input tensors shapes
  TensorShapeProtoList       m_OutputTensorsShapes;     // Output tensors shapes

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelBase.hxx"

#endif
