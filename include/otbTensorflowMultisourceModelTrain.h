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
#include "otbTensorflowMultisourceModelLearningBase.h"

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
 *
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage>
class ITK_EXPORT TensorflowMultisourceModelTrain :
public TensorflowMultisourceModelLearningBase<TInputImage>
{
public:

  /** Standard class typedefs. */
  typedef TensorflowMultisourceModelTrain                     Self;
  typedef TensorflowMultisourceModelLearningBase<TInputImage> Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowMultisourceModelTrain, TensorflowMultisourceModelLearningBase);

  /** Superclass typedefs */
  typedef typename Superclass::IndexValueType    IndexValueType;
  typedef typename Superclass::TensorListType    TensorListType;
  typedef typename Superclass::IndexListType     IndexListType;


protected:
  TensorflowMultisourceModelTrain();
  virtual ~TensorflowMultisourceModelTrain() {};

  void GenerateData();
  void ProcessBatch(TensorListType & inputs, const IndexValueType & sampleStart,
      const IndexValueType & batchSize);

private:
  TensorflowMultisourceModelTrain(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  IndexListType     m_RandomIndices;           // Reordered indices

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelTrain.hxx"

#endif
