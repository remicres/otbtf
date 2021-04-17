/*=========================================================================

  Copyright (c) 2018-2019 Remi Cresson (IRSTEA)
  Copyright (c) 2020-2021 Remi Cresson (INRAE)


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWSOURCE_HXX_
#define MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWSOURCE_HXX_

#include <otbTensorflowSource.h>

namespace otb
{

//
// Prepare the big stack of images
//
template <class TImage>
void
TensorflowSource<TImage>::Set(FloatVectorImageListType * inputList)
{
  // Create one stack for input images list
  m_Concatener    = ListConcatenerFilterType::New();
  m_List          = ImageListType::New();
  m_ExtractorList = ExtractROIFilterListType::New();

  // Split each input vector image into image
  // and generate an mono channel image list
  inputList->GetNthElement(0)->UpdateOutputInformation();
  SizeType size = inputList->GetNthElement(0)->GetLargestPossibleRegion().GetSize();
  for( unsigned int i = 0; i < inputList->Size(); i++ )
  {
    FloatVectorImagePointerType vectIm = inputList->GetNthElement(i);
    vectIm->UpdateOutputInformation();
    if( size != vectIm->GetLargestPossibleRegion().GetSize() )
    {
      itkGenericExceptionMacro("Input image size number " << i << " mismatch");
    }

    for( unsigned int j = 0; j < vectIm->GetNumberOfComponentsPerPixel(); j++)
    {
      typename MultiToMonoChannelFilterType::Pointer extractor = MultiToMonoChannelFilterType::New();
      extractor->SetInput( vectIm );
      extractor->SetChannel( j+1 );
      extractor->UpdateOutputInformation();
      m_ExtractorList->PushBack( extractor );
      m_List->PushBack( extractor->GetOutput() );
    }
  }
  m_Concatener->SetInput( m_List );
  m_Concatener->UpdateOutputInformation();

}

//
// Return the output image resulting from the big stack
//
template <class TImage>
typename TImage::Pointer
TensorflowSource<TImage>::Get()
{
  return m_Concatener->GetOutput();
}

} // end namespace otb

#endif
