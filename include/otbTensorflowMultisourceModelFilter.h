/*=========================================================================

  Copyright (c) 2018-2019 Remi Cresson (IRSTEA)
  Copyright (c) 2020-2021 Remi Cresson (INRAE)


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

// Tile hint
#include "itkMetaDataObject.h"
#include "otbMetaDataKey.h"

namespace otb
{

/**
 * \class TensorflowMultisourceModelFilter
 * \brief This filter apply a TensorFlow model over multiple input images and
 * generates one output image corresponding to outputs of the model.
 *
 * The filter takes N input images and feed the TensorFlow model to produce
 * one output image corresponding to the desired results of the TensorFlow model.
 * Names of input placeholders and output tensors must be specified using the
 * SetPlaceholders() and SetTensors() methods.
 *
 * Example: we have a TensorFlow model which runs the input images "x1" and "x2"
 *          and produces the output image "y".
 *          "x1" and "x2" are two placeholders, we set InputPlaceholder={"x1","x2"}
 *          "y1" corresponds to one output tensor, we set OutputTensors={"y1"}
 *
 * The filter can work in two modes:
 *
 * 1.Patch-based mode:
 *    Extract and process patches independently at regular intervals.
 *    Patches sizes are equal to the receptive field sizes of inputs. For each input,
 *    a tensor with a number of elements equal to the number of patches is fed to the
 *    TensorFlow model.
 *
 * 2.Fully-convolutional:
 *    Unlike patch-based mode, it allows the processing of an entire requested region.
 *    For each input, a tensor composed of one single element, corresponding to the input
 *    requested region, is fed to the TF model. This mode requires that receptive fields,
 *    expression fields and scale factors are consistent with operators implemented in the
 *    TensorFlow model, input images physical spacing and alignment.
 *    The filter produces output blocks avoiding any blocking artifact in fully-convolutional
 *    mode. This is done in computing input images regions that are aligned to the expression
 *    field sizes of the model (eventually, input requested regions are enlarged, but still
 *    aligned), and keeping only the subset of the output corresponding to the requested
 *    output region.
 *
 * The reference grid for the output image is the same as the first input image.
 * This grid can be scaled by setting the OutputSpacingScale value.
 * This can be used to run models which downsize the output image spacing
 * (e.g. fully convolutional model with strides) or to produce the result
 * of a patch-based network at regular intervals.
 *
 * For each input (resp. output), receptive field (resp. expression field) must be set.
 * If the number of values in the output tensors (produced by the model) don't
 * fit with the output image region, an exception will be thrown.
 *
 *
 * TODO: the filter must be able to output multiple images eventually at different
 * resolutions/sizes/origins.
 *
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
  typedef typename Superclass::DictElementType     DictElementType;
  typedef typename Superclass::DictType            DictType;
  typedef typename Superclass::StringList          StringList;
  typedef typename Superclass::SizeListType        SizeListType;
  typedef typename Superclass::TensorListType      TensorListType;
  typedef std::vector<float>                       ScaleListType;

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

  SizeType                   m_OutputGridSize;       // Output grid size
  bool                       m_ForceOutputGridSize;  // Force output grid size
  bool                       m_FullyConvolutional;   // Convolution mode
  float                      m_OutputSpacingScale;   // scaling of the output spacings

  // Internal
  SpacingType                m_OutputSpacing;     // Output image spacing
  PointType                  m_OutputOrigin;      // Output image origin
  SizeType                   m_OutputSize;        // Output image size
  PixelType                  m_NullPixel;         // Pixel filled with zeros

}; // end class


} // end namespace otb

#include "otbTensorflowMultisourceModelFilter.hxx"

#endif
