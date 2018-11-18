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
 * \brief This filter is the base class for all filters that input patches images.
 *
 * One input patches image consist in an image of size (pszx, pszy*n, nbands) where:
 * -pszx   : is the width of one patch
 * -pszy   : is the height of one patch
 * -n      : is the number of patches in the patches image
 * -nbands : is the number of channels in the patches image
 *
 * This filter verify that every patches images are consistent.
 *
 * The batch size can be set using the SetBatchSize() method.
 * The streaming can be activated to allow the processing of huge datasets.
 * However, it should be noted that the process is significantly slower due to
 * multiple read of input patches. When streaming is deactivated, the whole
 * patches images are read and kept in memory, guaranteeing fast patches access.
 *
 * The GenerateData() implements a loop over batches, that call the ProcessBatch()
 * method for each one.
 * The ProcessBatch() function is a pure virtual method that must be implemented in
 * child classes.
 *
 * The PopulateInputTensors() method converts input patches images into placeholders
 * that will be fed to the model. It is a common method to learning filters, and
 * is intended to be used in child classes, as a kind of helper.
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
