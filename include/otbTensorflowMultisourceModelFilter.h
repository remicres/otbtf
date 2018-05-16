/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelFilter_h
#define otbTensorflowMultisourceModelFilter_h

#include "otbTensorflowMultisourceModelBase.h"

// Iterator
#include "itkImageRegionConstIteratorWithOnlyIndex.h"

// Tensorflow helpers
#include "otbTensorflowGraphOperations.h"
#include "otbTensorflowDataTypeBridge.h"
#include "otbTensorflowCopyUtils.h"

namespace otb
{

/**
 * \class TensorflowMultisourceModelFilter
 * \brief This filter apply a TensorFlow model over multiple input images.
 *
 * The filter takes N input images and feed the TensorFlow model to produce
 * one output image of desired TF op results.
 * Names of input/output placeholders/tensors must be specified using the
 * SetInputPlaceholdersNames/SetOutputTensorNames.
 *
 * Example: we have a tensorflow model which runs the input images "x1" and "x2"
 *          and produces the output image "y".
 *          "x1" and "x2" are two TF placeholders, we set InputTensorNames={"x1","x2"}
 *          "y1" corresponds to one TF op output, we set OutputTensorNames={"y1"}
 *
 * The reference grid for the output image is the same as the first input image.
 * This grid can be scaled by setting the OutputSpacingScale value.
 * This can be used to run models which downsize the output image spacing
 * (typically fully convolutional model with strides) or to produce the result
 * of a patch-based network at regular intervals.
 *
 * For each input, input field of view (FOV) must be set.
 * If the number of values in the output tensors (produced by the model) don't
 * fit with the output image region, exception will be thrown.
 *
 *
 * The tensorflow Graph is passed using the SetGraph() method
 * The tensorflow Session is passed using the SetSession() method
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT TensorflowMultisourceModelFilter :
public TensorflowMultisourceModelBase<TInputImage, TOutputImage>
{

public:

  /** Standard class typedefs. */
  typedef TensorflowMultisourceModelFilter                          Self;
  typedef TensorflowMultisourceModelBase<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                   Pointer;
  typedef itk::SmartPointer<const Self>                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowMultisourceModelFilter, TensorflowMultisourceModelBase);

  /** Images typedefs */
  typedef typename Superclass::ImageType           ImageType;
  typedef typename Superclass::ImagePointerType    ImagePointerType;
  typedef typename Superclass::PixelType           PixelType;
  typedef typename Superclass::IndexType           IndexType;
  typedef typename IndexType::IndexValueType       IndexValueType;
  typedef typename Superclass::PointType           PointType;
  typedef typename Superclass::SizeType            SizeType;
  typedef typename SizeType::SizeValueType         SizeValueType;
  typedef typename Superclass::SpacingType         SpacingType;
  typedef typename Superclass::RegionType          RegionType;

  typedef TOutputImage                             OutputImageType;
  typedef typename TOutputImage::PixelType         OutputPixelType;
  typedef typename TOutputImage::InternalPixelType OutputInternalPixelType;

  /* Iterators typedefs */
  typedef typename itk::ImageRegionConstIteratorWithOnlyIndex<TInputImage> IndexIteratorType;
  typedef typename itk::ImageRegionConstIterator<TInputImage>              InputConstIteratorType;

  /* Typedefs for parameters */
  typedef typename Superclass::DictType            DictType;
  typedef typename Superclass::StringList          StringList;
  typedef typename Superclass::SizeListType        SizeListType;
  typedef typename Superclass::DictListType        DictListType;
  typedef typename Superclass::TensorListType      TensorListType;
  typedef std::vector<float>                       ScaleListType;

  itkSetMacro(OutputFOESize, SizeType);
  itkGetMacro(OutputFOESize, SizeType);
  itkSetMacro(OutputGridSize, SizeType);
  itkGetMacro(OutputGridSize, SizeType);
  itkSetMacro(ForceOutputGridSize, bool);
  itkGetMacro(ForceOutputGridSize, bool);
  itkSetMacro(FullyConvolutional, bool);
  itkGetMacro(FullyConvolutional, bool);
  itkSetMacro(OutputSpacingScale, float);
  itkGetMacro(OutputSpacingScale, float);

protected:
  TensorflowMultisourceModelFilter();
  virtual ~TensorflowMultisourceModelFilter() {};

  virtual void SmartPad(RegionType& region, const SizeType &patchSize);
  virtual void SmartShrink(RegionType& region, const SizeType &patchSize);
  virtual void ImageToExtent(ImageType* image, PointType &extentInf, PointType &extentSup, SizeType &patchSize);
  virtual bool OutputRegionToInputRegion(const RegionType &outputRegion, RegionType &inputRegion, ImageType* &inputImage);
  virtual void EnlargeToAlignedRegion(RegionType& region);

  virtual void GenerateOutputInformation(void);

  virtual void GenerateInputRequestedRegion(void);

  virtual void GenerateData();

private:
  TensorflowMultisourceModelFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  SizeType                   m_OutputFOESize;        // Output tensors field of expression (FOE) sizes
  SizeType                   m_OutputGridSize;       // Output grid size
  bool                       m_ForceOutputGridSize;  // Force output grid size
  bool                       m_FullyConvolutional;   // Convolution mode
  float                      m_OutputSpacingScale;   // scaling of the output spacings

  // Internal
  SpacingType                m_OutputSpacing;     // Output image spacing
  PointType                  m_OutputOrigin;      // Output image origin
  SizeType                   m_OutputSize;        // Output image size

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelFilter.hxx"

#endif
