/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelValidate_h
#define otbTensorflowMultisourceModelValidate_h

#include "itkProcessObject.h"
#include "itkNumericTraits.h"
#include "itkSimpleDataObjectDecorator.h"

// Base
#include "otbTensorflowMultisourceModelBase.h"

// Iterate over images
#include "otbTensorflowCommon.h"
#include "itkImageRegionConstIterator.h"

// Matrix
#include "itkVariableSizeMatrix.h"

namespace otb
{

/**
 * \class TensorflowMultisourceModelValidate
 * \brief This filter validates a TensorFlow model over multiple input images.
 *
 * This filter computes confusion matrices for each output tensor.
 * The references (i.e. ground truth for validation) must be set using the
 * SetReferences() method. References must be provided in the same order as
 * their related output tensors (i.e. names and patch sizes). If the number of
 * references is not the same as output tensors, an exception is thrown.
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage>
class ITK_EXPORT TensorflowMultisourceModelValidate :
public TensorflowMultisourceModelLearningBase<TInputImage>
{
public:

  /** Standard class typedefs. */
  typedef TensorflowMultisourceModelValidate                  Self;
  typedef TensorflowMultisourceModelLearningBase<TInputImage> Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowMultisourceModelValidate, TensorflowMultisourceModelLearningBase);

  /** Images typedefs */
  typedef typename Superclass::ImageType         ImageType;
  typedef typename Superclass::ImagePointerType  ImagePointerType;
  typedef typename Superclass::RegionType        RegionType;
  typedef typename Superclass::SizeType          SizeType;
  typedef typename Superclass::IndexType         IndexType;
  typedef std::vector<ImagePointerType>          ImageListType;

  /* Typedefs for parameters */
  typedef typename Superclass::DictType          DictType;
  typedef typename Superclass::StringList        StringList;
  typedef typename Superclass::SizeListType      SizeListType;
  typedef typename Superclass::TensorListType    TensorListType;
  typedef typename Superclass::IndexValueType    IndexValueType;
  typedef typename Superclass::IndexListType     IndexListType;

  /* Typedefs for validation */
  typedef unsigned long                            CountValueType;
  typedef int                                      LabelValueType;
  typedef std::map<LabelValueType, CountValueType> MapType;
  typedef std::map<LabelValueType, MapType>        MatMapType;
  typedef std::map<LabelValueType, LabelValueType> MapOfClassesType;
  typedef std::vector<MapOfClassesType>            MapOfClassesListType;
  typedef itk::VariableSizeMatrix<CountValueType>  ConfMatType;
  typedef std::vector<ConfMatType>                 ConfMatListType;
  typedef itk::ImageRegionConstIterator<ImageType> IteratorType;

  /** Set and Get the input references */
  virtual void SetInputReferences(ImageListType input);
  ImagePointerType GetInputReference(unsigned int index);

  /** Get the confusion matrix */
  const ConfMatType GetConfusionMatrix(unsigned int target);

  /** Get the map of classes matrix */
  const MapOfClassesType GetMapOfClasses(unsigned int target);

protected:
  TensorflowMultisourceModelValidate();
  virtual ~TensorflowMultisourceModelValidate() {};

  void GenerateOutputInformation(void);
  void GenerateData();
  void ProcessBatch(DictType & inputs, const IndexValueType & sampleStart,
      const IndexValueType & batchSize);

private:
  TensorflowMultisourceModelValidate(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  ImageListType              m_References;              // The references images

  // Read only
  ConfMatListType            m_ConfusionMatrices;       // Confusion matrix
  MapOfClassesListType       m_MapsOfClasses;           // Maps of classes

  // Internal
  std::vector<MatMapType>    m_ConfMatMaps;             // Accumulators

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelValidate.hxx"

#endif
