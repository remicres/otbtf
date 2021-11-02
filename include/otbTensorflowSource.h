/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWSOURCE_H_
#define MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWSOURCE_H_

#include "otbImage.h"
#include "otbImageListToVectorImageFilter.h"
#include "otbMultiToMonoChannelExtractROI.h"
#include "otbImageList.h"
#include "otbMultiChannelExtractROI.h"
#include "otbExtractROI.h"

#include "otbTensorflowCommon.h"

namespace otb
{

/*
 * This is a simple helper to create images concatenation.
 * Images must have the same size.
 * This is basically the common input type used in every OTB-TF applications.
 */
template<class TImage>
class TensorflowSource
{
public:
  /** Typedefs for images */
  typedef TImage                                            FloatVectorImageType;
  typedef typename FloatVectorImageType::Pointer            FloatVectorImagePointerType;
  typedef typename FloatVectorImageType::InternalPixelType  InternalPixelType;
  typedef otb::Image<InternalPixelType>                     FloatImageType;
  typedef typename FloatImageType::SizeType                 SizeType;

  /** Typedefs for image concatenation */
  typedef otb::ImageList<FloatImageType>                    ImageListType;
  typedef typename ImageListType::Pointer                   ImageListPointer;
  typedef ImageListToVectorImageFilter<ImageListType,
      FloatVectorImageType>                                 ListConcatenerFilterType;
  typedef typename ListConcatenerFilterType::Pointer        ListConcatenerFilterPointer;
  typedef MultiToMonoChannelExtractROI<InternalPixelType,
      InternalPixelType>                                    MultiToMonoChannelFilterType;
  typedef ObjectList<MultiToMonoChannelFilterType>          ExtractROIFilterListType;
  typedef typename ExtractROIFilterListType::Pointer        ExtractROIFilterListPointer;
  typedef otb::MultiChannelExtractROI<InternalPixelType,
      InternalPixelType>                                    ExtractFilterType;
  typedef otb::ObjectList<FloatVectorImageType>             FloatVectorImageListType;

  // Initialize the source
  void Set(FloatVectorImageListType * inputList);

  // Get the source output
  FloatVectorImagePointerType Get();

  TensorflowSource(){};
  virtual ~TensorflowSource (){};

private:
  ListConcatenerFilterPointer m_Concatener;    // Mono-images stacker
  ImageListPointer            m_List;          // List of mono-images
  ExtractROIFilterListPointer m_ExtractorList; // Mono-images extractors

};

} // end namespace otb

#include "otbTensorflowSource.hxx"

#endif /* MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWSOURCE_H_ */
