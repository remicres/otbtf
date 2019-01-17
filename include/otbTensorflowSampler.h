/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowSampler_h
#define otbTensorflowSampler_h

#include "itkProcessObject.h"
#include "itkNumericTraits.h"
#include "itkSimpleDataObjectDecorator.h"

// Extract ROI
#include "otbMultiChannelExtractROI.h"

// TF common
#include "otbTensorflowCommon.h"

// Tree iterator
#include "itkPreOrderTreeIterator.h"

namespace otb
{

/**
 * \class TensorflowSampler
 * \brief This process objects performs samples extraction from an input image.
 *
 * The filter takes one input image and extract samples of fixed size.
 * Samples are concatenated in y dimension to form a single big image of
 * extracted patches.
 * Label image is also created from the value of the m_Field field of the
 * input vector data
 *
 * TODO:
 * -must inherit from itk::imageToImageFilter
 * -implement streaming mechanism : the input requested region of
 *  image should be computed from the output requested region of the patches
 *  This would allow to compute huge patches images and speed up the whole
 *  process. This might be achieved using indexation structure like RTree
 *  on the samples pos (in image coordinates)
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage, class TVectorData>
class ITK_EXPORT TensorflowSampler :
public itk::ProcessObject
{
public:

  /** Standard class typedefs. */
  typedef TensorflowSampler                       Self;
  typedef itk::ProcessObject                      Superclass;
  typedef itk::SmartPointer<Self>                 Pointer;
  typedef itk::SmartPointer<const Self>           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowSampler, itk::ProcessObject);

  /** Images typedefs */
  typedef TInputImage                             ImageType;
  typedef typename TInputImage::Pointer           ImagePointerType;
  typedef typename TInputImage::InternalPixelType InternalPixelType;
  typedef typename TInputImage::PixelType         PixelType;
  typedef typename TInputImage::RegionType        RegionType;
  typedef typename TInputImage::PointType         PointType;
  typedef typename TInputImage::SizeType          SizeType;
  typedef typename TInputImage::IndexType         IndexType;
  typedef typename otb::MultiChannelExtractROI<InternalPixelType,
      InternalPixelType>                          ExtractROIMultiFilterType;
  typedef typename ExtractROIMultiFilterType::Pointer
                                                  ExtractROIMultiFilterPointerType;
  typedef typename std::vector<ImagePointerType>  ImagePointerListType;
  typedef typename std::vector<SizeType>          SizeListType;

  /** Vector data typedefs */
  typedef TVectorData                             VectorDataType;
  typedef typename VectorDataType::Pointer        VectorDataPointer;
  typedef typename VectorDataType::DataTreeType   DataTreeType;
  typedef typename itk::PreOrderTreeIterator<DataTreeType>
                                                  TreeIteratorType;
  typedef typename VectorDataType::DataNodeType   DataNodeType;
  typedef typename DataNodeType::Pointer          DataNodePointer;
  typedef typename DataNodeType::PolygonListPointerType
                                                  PolygonListPointerType;

  /** Set / get parameters */
  itkSetMacro(Field, std::string);
  itkGetMacro(Field, std::string);

  /** Set / get vector data */
  itkSetMacro(InputVectorData, VectorDataPointer);
  itkGetConstMacro(InputVectorData, VectorDataPointer);

  /** Set / get image */
  virtual void PushBackInputWithPatchSize(const ImageType *input, SizeType & patchSize);
  const ImageType* GetInput(unsigned int index);

  /** Do the real work */
  virtual void Update();

  /** Get outputs */
  itkGetMacro(OutputPatchImages, ImagePointerListType);
  itkGetMacro(OutputLabelImage, ImagePointerType);
  itkGetMacro(NumberOfAcceptedSamples, unsigned long);
  itkGetMacro(NumberOfRejectedSamples, unsigned long);

protected:
  TensorflowSampler();
  virtual ~TensorflowSampler() {};

  virtual void ResizeImage(ImagePointerType & image, SizeType & patchSize, unsigned int nbSamples);
  virtual void AllocateImage(ImagePointerType & image, SizeType & patchSize, unsigned int nbSamples, unsigned int nbComponents);

private:
  TensorflowSampler(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  std::string          m_Field;
  SizeListType         m_PatchSizes;
  VectorDataPointer    m_InputVectorData;

  // Read only
  ImagePointerListType m_OutputPatchImages;
  ImagePointerType     m_OutputLabelImage;
  unsigned long        m_NumberOfAcceptedSamples;
  unsigned long        m_NumberOfRejectedSamples;

}; // end class

} // end namespace otb

#include "otbTensorflowSampler.hxx"

#endif
