/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowStreamerFilter_h
#define otbTensorflowStreamerFilter_h

#include "itkImageToImageFilter.h"
#include "itkProgressReporter.h"

namespace otb
{

/**
 * \class TensorflowStreamerFilter
 * \brief This filter generates an output image with an internal
 * explicit streaming mechanism.
 *
 * \ingroup OTBTensorflow
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT TensorflowStreamerFilter :
public itk::ImageToImageFilter<TInputImage, TOutputImage>
{

public:

  /** Standard class typedefs. */
  typedef TensorflowStreamerFilter                           Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TensorflowStreamerFilter, itk::ImageToImageFilter);

  /** Images typedefs */
  typedef typename Superclass::InputImageType       ImageType;
  typedef typename ImageType::IndexType             IndexType;
  typedef typename ImageType::IndexValueType        IndexValueType;
  typedef typename ImageType::SizeType              SizeType;
  typedef typename Superclass::InputImageRegionType RegionType;

  typedef TOutputImage                             OutputImageType;

  itkSetMacro(OutputGridSize, SizeType);
  itkGetMacro(OutputGridSize, SizeType);

protected:
  TensorflowStreamerFilter();
  virtual ~TensorflowStreamerFilter() {};

  virtual void UpdateOutputData(itk::DataObject *output){(void) output; this->GenerateData();}

  virtual void GenerateData();

private:
  TensorflowStreamerFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  SizeType                   m_OutputGridSize;       // Output grid size

}; // end class


} // end namespace otb

#include "otbTensorflowStreamerFilter.hxx"

#endif
