/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelTrain_h
#define otbTensorflowMultisourceModelTrain_h

#include "itkProcessObject.h"
#include "itkNumericTraits.h"
#include "itkSimpleDataObjectDecorator.h"

// Base
#include "otbTensorflowMultisourceModelBase.h"

// Shuffle
#include <random>
#include <algorithm>
#include <iterator>

namespace otb
{

/**
 * \class TensorflowMultisourceModelTrain
 * \brief This filter train a TensorFlow model over multiple input images.
 *
 * The filter takes N input images and feed the TensorFlow model.
 * Names of input placeholders must be specified using the
 * SetInputPlaceholdersNames method
 *
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage>
class ITK_EXPORT TensorflowMultisourceModelTrain :
public TensorflowMultisourceModelBase<TInputImage>
{
public:

  /** Standard class typedefs. */
  typedef TensorflowMultisourceModelTrain                    Self;
  typedef TensorflowMultisourceModelBase<TInputImage>        Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowMultisourceModelTrain, TensorflowMultisourceModelBase);

  /** Images typedefs */
  typedef typename Superclass::ImageType         ImageType;
  typedef typename Superclass::ImagePointerType  ImagePointerType;
  typedef typename Superclass::RegionType        RegionType;
  typedef typename Superclass::SizeType          SizeType;
  typedef typename Superclass::IndexType         IndexType;

  /* Typedefs for parameters */
  typedef typename Superclass::DictType          DictType;
  typedef typename Superclass::StringList        StringList;
  typedef typename Superclass::SizeListType      SizeListType;
  typedef typename Superclass::DictListType      DictListType;
  typedef typename Superclass::TensorListType    TensorListType;

  itkSetMacro(BatchSize, unsigned int);
  itkGetMacro(BatchSize, unsigned int);
  itkGetMacro(NumberOfSamples, unsigned int);

  virtual void GenerateOutputInformation(void);

  virtual void GenerateInputRequestedRegion();

  virtual void GenerateData();

protected:
  TensorflowMultisourceModelTrain();
  virtual ~TensorflowMultisourceModelTrain() {};

private:
  TensorflowMultisourceModelTrain(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  unsigned int               m_BatchSize;               // Batch size

  // Read only
  unsigned int               m_NumberOfSamples;         // Number of samples

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelTrain.hxx"

#endif
