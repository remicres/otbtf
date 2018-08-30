/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelLearningBase_h
#define otbTensorflowMultisourceModelLearningBase_h

#include "itkProcessObject.h"
#include "itkNumericTraits.h"
#include "itkSimpleDataObjectDecorator.h"

// Base
#include "otbTensorflowMultisourceModelBase.h"

namespace otb
{

/**
 * \class TensorflowMultisourceModelLearningBase
 * \brief This filter is the base class for learning filters.
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage>
class ITK_EXPORT TensorflowMultisourceModelLearningBase :
public TensorflowMultisourceModelBase<TInputImage>
{
public:

  /** Standard class typedefs. */
  typedef TensorflowMultisourceModelLearningBase       Self;
  typedef TensorflowMultisourceModelBase<TInputImage>  Superclass;
  typedef itk::SmartPointer<Self>                      Pointer;
  typedef itk::SmartPointer<const Self>                ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowMultisourceModelLearningBase, TensorflowMultisourceModelBase);

  /** Images typedefs */
  typedef typename Superclass::ImageType         ImageType;
  typedef typename Superclass::ImagePointerType  ImagePointerType;
  typedef typename Superclass::RegionType        RegionType;
  typedef typename Superclass::SizeType          SizeType;
  typedef typename Superclass::IndexType         IndexType;

  /* Typedefs for parameters */
  typedef typename Superclass::DictType          DictType;
  typedef typename Superclass::DictElementType   DictElementType;
  typedef typename Superclass::StringList        StringList;
  typedef typename Superclass::SizeListType      SizeListType;
  typedef typename Superclass::TensorListType    TensorListType;

  /* Typedefs for index */
  typedef typename ImageType::IndexValueType     IndexValueType;
  typedef std::vector<IndexValueType>            IndexListType;

  // Batch size
  itkSetMacro(BatchSize, IndexValueType);
  itkGetMacro(BatchSize, IndexValueType);

  // Use streaming
  itkSetMacro(UseStreaming, bool);
  itkGetMacro(UseStreaming, bool);

  // Get number of samples
  itkGetMacro(NumberOfSamples, IndexValueType);

protected:
  TensorflowMultisourceModelLearningBase();
  virtual ~TensorflowMultisourceModelLearningBase() {};

  virtual void GenerateOutputInformation(void);

  virtual void GenerateInputRequestedRegion();

  virtual void GenerateData();

  virtual void PopulateInputTensors(DictType & inputs, const IndexValueType & sampleStart,
      const IndexValueType & batchSize, const IndexListType & order);

  virtual void ProcessBatch(DictType & inputs, const IndexValueType & sampleStart,
      const IndexValueType & batchSize) = 0;

private:
  TensorflowMultisourceModelLearningBase(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  unsigned int          m_BatchSize;       // Batch size
  bool                  m_UseStreaming;    // Use streaming on/off

  // Read only
  IndexValueType        m_NumberOfSamples; // Number of samples

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelLearningBase.hxx"

#endif
