/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


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
 * The filter takes N input images and feed the TensorFlow model.
 * Names of input placeholders must be specified using the
 * SetInputPlaceholdersNames method
 *
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage>
class ITK_EXPORT TensorflowMultisourceModelValidate :
public TensorflowMultisourceModelBase<TInputImage>
{
public:

  /** Standard class typedefs. */
  typedef TensorflowMultisourceModelValidate                    Self;
  typedef TensorflowMultisourceModelBase<TInputImage>        Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowMultisourceModelValidate, TensorflowMultisourceModelBase);

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
  typedef typename Superclass::DictListType      DictListType;
  typedef typename Superclass::TensorListType    TensorListType;

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

  itkSetMacro(BatchSize, unsigned int);
  itkGetMacro(BatchSize, unsigned int);
  itkGetMacro(NumberOfSamples, unsigned int);

  virtual void GenerateOutputInformation(void);

  virtual void GenerateInputRequestedRegion();

  virtual void SetInputReferences(ImageListType input);
  ImagePointerType GetInputReference(unsigned int index);

  virtual void GenerateData();

  const ConfMatType GetConfusionMatrix(unsigned int target);
  const MapOfClassesType GetMapOfClasses(unsigned int target);

protected:
  TensorflowMultisourceModelValidate();
  virtual ~TensorflowMultisourceModelValidate() {};

private:
  TensorflowMultisourceModelValidate(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  unsigned int               m_BatchSize;               // Batch size
  ImageListType              m_References;              // The references images

  // Read only
  unsigned int               m_NumberOfSamples;         // Number of samples
  ConfMatListType            m_ConfusionMatrices;       // Confusion matrix
  MapOfClassesListType       m_MapsOfClasses;           // Maps of classes

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelValidate.hxx"

#endif
