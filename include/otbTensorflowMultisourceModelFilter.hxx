/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbTensorflowMultisourceModelFilter_txx
#define otbTensorflowMultisourceModelFilter_txx

#include "otbTensorflowMultisourceModelFilter.h"

namespace otb
{

template <class TInputImage, class TOutputImage>
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::TensorflowMultisourceModelFilter()
 {
  m_OutputGridSize.Fill(0);
  m_ForceOutputGridSize = false;
  m_FullyConvolutional = false;

  m_OutputSpacing.Fill(0);
  m_OutputOrigin.Fill(0);
  m_OutputSize.Fill(0);

  m_OutputSpacingScale = 1.0f;

  Superclass::SetCoordinateTolerance(itk::NumericTraits<double>::max() );
  Superclass::SetDirectionTolerance(itk::NumericTraits<double>::max() );
 }

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::SmartPad(RegionType& region, const SizeType &patchSize)
 {
  for(unsigned int dim = 0; dim<OutputImageType::ImageDimension; ++dim)
    {
    const SizeValueType psz = patchSize[dim];
    const SizeValueType rval = 0.5 * psz;
    const SizeValueType lval = psz - rval;
    region.GetModifiableIndex()[dim] -= lval;
    region.GetModifiableSize()[dim] += psz;
    }
 }

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::SmartShrink(RegionType& region, const SizeType &patchSize)
 {
  for(unsigned int dim = 0; dim<OutputImageType::ImageDimension; ++dim)
    {
    const SizeValueType psz = patchSize[dim];
    const SizeValueType lval = 0.5 * psz;
    region.GetModifiableIndex()[dim] += lval;
    region.GetModifiableSize()[dim] -= psz - 1;
    }
 }

/**
  Compute the input image extent: corners inf and sup.
  Very important: **corners**, not pixel center
 */
template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::ImageToExtent(ImageType* image, PointType &extentInf, PointType &extentSup, SizeType &patchSize)
 {

  // Get largest possible region
  RegionType largestPossibleRegion = image->GetLargestPossibleRegion();

  // Shrink it a little with the FOV radius
  SmartShrink(largestPossibleRegion, patchSize);

  // Get index of first and last pixel
  IndexType imageFirstIndex = largestPossibleRegion.GetIndex();
  IndexType imageLastIndex = largestPossibleRegion.GetUpperIndex();

  // Compute extent
  PointType imageOrigin;
  PointType imageEnd;
  image->TransformIndexToPhysicalPoint(imageLastIndex, imageEnd);
  image->TransformIndexToPhysicalPoint(imageFirstIndex, imageOrigin);
  for(unsigned int dim = 0; dim<OutputImageType::ImageDimension; ++dim)
    {
    extentInf[dim] = vnl_math_min(imageOrigin[dim], imageEnd[dim]) - 0.5 * image->GetSpacing()[dim];
    extentSup[dim] = vnl_math_max(imageOrigin[dim], imageEnd[dim]) + 0.5 * image->GetSpacing()[dim];
    }

 }

/**
  Compute the region of the input image which correspond to the given output requested region
  Return true if the region exists, false if not
  Function taken from "Mosaic"
 */
template <class TInputImage, class TOutputImage>
bool
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::OutputRegionToInputRegion(const RegionType &outputRegion, RegionType &inputRegion, ImageType* &inputImage)
 {

  // Mosaic Region Start & End (mosaic image index)
  const IndexType outIndexStart = outputRegion.GetIndex();
  const IndexType outIndexEnd = outputRegion.GetUpperIndex();

  // Mosaic Region Start & End (geo)
  PointType outPointStart, outPointEnd;
  this->GetOutput()->TransformIndexToPhysicalPoint(outIndexStart, outPointStart);
  this->GetOutput()->TransformIndexToPhysicalPoint(outIndexEnd  , outPointEnd  );

  // Add the half-width pixel size of the input image
  // and remove the half-width pixel size of the output image
  // (coordinates = pixel center)
  const SpacingType outputSpc = this->GetOutput()->GetSpacing();
  const SpacingType inputSpc = inputImage->GetSpacing();
  for(unsigned int dim = 0; dim<OutputImageType::ImageDimension; ++dim)
    {
    const typename SpacingType::ValueType border =
        0.5 * (inputSpc[dim] - outputSpc[dim]);
    if (outPointStart[dim] < outPointEnd[dim])
      {
      outPointStart[dim] += border;
      outPointEnd  [dim] -= border;
      }
    else
      {
      outPointStart[dim] -= border;
      outPointEnd  [dim] += border;
      }
    }

  // Mosaic Region Start & End (input image index)
  IndexType defIndexStart, defIndexEnd;
  inputImage->TransformPhysicalPointToIndex(outPointStart, defIndexStart);
  inputImage->TransformPhysicalPointToIndex(outPointEnd  , defIndexEnd);

  // Compute input image region
  for(unsigned int dim = 0; dim<OutputImageType::ImageDimension; ++dim)
    {
    inputRegion.SetIndex(dim, vnl_math_min(defIndexStart[dim], defIndexEnd[dim]));
    inputRegion.SetSize(dim, vnl_math_max(defIndexStart[dim], defIndexEnd[dim]) - inputRegion.GetIndex(dim) + 1);
    }

  // crop the input requested region at the input's largest possible region
  return inputRegion.Crop( inputImage->GetLargestPossibleRegion() );

 }

/*
 * Enlarge the given region to the nearest aligned region.
 * Aligned region = Index and UpperIndex+1 are on the output grid
 */
template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::EnlargeToAlignedRegion(RegionType& region)
 {
  for(unsigned int dim = 0; dim<OutputImageType::ImageDimension; ++dim)
    {
    // Get corners
    IndexValueType lower = region.GetIndex(dim);
    IndexValueType upper = lower + region.GetSize(dim);

    // Compute deltas between corners and the grid
    const IndexValueType deltaLo = lower % m_OutputGridSize[dim];
    const IndexValueType deltaUp = upper % m_OutputGridSize[dim];

    // Move corners to aligned positions
    lower -= deltaLo;
    if (deltaUp > 0)
      {
      upper += m_OutputGridSize[dim] - deltaUp;
      }

    // Update region
    region.SetIndex(dim, lower);
    region.SetSize(dim, upper - lower);

    }
 }

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
 {

  Superclass::GenerateOutputInformation();

  //////////////////////////////////////////////////////////////////////////////////////////
  //                            Compute the output image extent
  //////////////////////////////////////////////////////////////////////////////////////////

  // If the output spacing is not specified, we use the first input image as grid reference
  // OTBTF assumes that the output image has the following geometric properties:
  // (1) Image origin is the top-left pixel
  // (2) Image pixel spacing has positive x-spacing and negative y-spacing
  m_OutputSpacing = this->GetInput(0)->GetSpacing();  // GetSpacing() returns abs. spacing
  m_OutputSpacing[1] *= -1.0;  // Force negative y-spacing
  m_OutputSpacing[0] *= m_OutputSpacingScale;
  m_OutputSpacing[1] *= m_OutputSpacingScale;

  // Compute the extent of each input images and update the extent or the output image.
  // The extent of the output image is the intersection of all input images extents.
  PointType extentInf, extentSup;
  extentSup.Fill(itk::NumericTraits<double>::max());
  extentInf.Fill(itk::NumericTraits<double>::NonpositiveMin());
  for (unsigned int imageIndex = 0 ; imageIndex < this->GetNumberOfInputs() ; imageIndex++)
    {
    ImageType * currentImage = static_cast<ImageType *>(
        Superclass::ProcessObject::GetInput(imageIndex) );

    // Update output image extent
    PointType currentInputImageExtentInf, currentInputImageExtentSup;
    ImageToExtent(currentImage, currentInputImageExtentInf, currentInputImageExtentSup, this->GetInputReceptiveFields()[imageIndex]);
    for(unsigned int dim = 0; dim<ImageType::ImageDimension; ++dim)
      {
      extentInf[dim] = vnl_math_max(currentInputImageExtentInf[dim], extentInf[dim]);
      extentSup[dim] = vnl_math_min(currentInputImageExtentSup[dim], extentSup[dim]);
      }
    }


  // Set final origin, aligned to the reference image grid.
  // Here we simply get back to the center of the pixel (extents are pixels corners coordinates)
  m_OutputOrigin[0] =  extentInf[0] + 0.5 * this->GetInput(0)->GetSpacing()[0];
  m_OutputOrigin[1] =  extentSup[1] - 0.5 * this->GetInput(0)->GetSpacing()[1];

  // Set final size
  m_OutputSize[0] = std::floor( (extentSup[0] - extentInf[0]) / std::abs(m_OutputSpacing[0]) );
  m_OutputSize[1] = std::floor( (extentSup[1] - extentInf[1]) / std::abs(m_OutputSpacing[1]) );

  // We should take in account one more thing: the expression field. It enlarge slightly the output image extent.
  m_OutputOrigin[0] -= m_OutputSpacing[0] * std::floor(0.5 * this->GetOutputExpressionFields().at(0)[0]);
  m_OutputOrigin[1] -= m_OutputSpacing[1] * std::floor(0.5 * this->GetOutputExpressionFields().at(0)[1]);
  m_OutputSize[0] += this->GetOutputExpressionFields().at(0)[0] - 1;
  m_OutputSize[1] += this->GetOutputExpressionFields().at(0)[1] - 1;

  // Set output grid size
  if (!m_ForceOutputGridSize)
    {
    // Default is the output field of expression
    m_OutputGridSize = this->GetOutputExpressionFields().at(0);
    }

  // Resize the largestPossibleRegion to be a multiple of the grid size
  for(unsigned int dim = 0; dim<ImageType::ImageDimension; ++dim)
    {
    if (m_OutputGridSize[dim] > m_OutputSize[dim])
      itkGenericExceptionMacro("Output grid size is larger than output image size !");
    m_OutputSize[dim] -= m_OutputSize[dim] % m_OutputGridSize[dim];
    }

  // Set the largest possible region
  RegionType largestPossibleRegion;
  largestPossibleRegion.SetSize(m_OutputSize);

  //////////////////////////////////////////////////////////////////////////////////////////
  //                  Compute the output number of components per pixel
  //////////////////////////////////////////////////////////////////////////////////////////

  unsigned int outputPixelSize = 0;
  for (auto& protoShape: this->GetOutputTensorsShapes())
    {
    // The number of components per pixel is the last dimension of the tensor
    int dim_size = protoShape.dim_size();
    unsigned int nComponents = 1;
    if (1 < dim_size && dim_size <= 4)
      {
      nComponents = protoShape.dim(dim_size-1).size();
      }
    else if (dim_size > 4)
      {
      itkExceptionMacro("Dim_size=" << dim_size << " currently not supported.");
      }
    outputPixelSize += nComponents;
    }

  // Copy input image projection
  ImageType * inputImage = static_cast<ImageType * >( Superclass::ProcessObject::GetInput(0) );
  const std::string projectionRef = inputImage->GetProjectionRef();

  // Set output image origin/spacing/size/projection
  ImageType * outputPtr = this->GetOutput();
  outputPtr->SetNumberOfComponentsPerPixel(outputPixelSize);
  outputPtr->SetProjectionRef        ( projectionRef );
  outputPtr->SetOrigin               ( m_OutputOrigin );
  outputPtr->SetSignedSpacing        ( m_OutputSpacing );
  outputPtr->SetLargestPossibleRegion( largestPossibleRegion );

  // Set null pixel
  m_NullPixel.SetSize(outputPtr->GetNumberOfComponentsPerPixel());
  m_NullPixel.Fill(0);

  //////////////////////////////////////////////////////////////////////////////////////////
  //                        Set the tiling layout hint in metadata
  //////////////////////////////////////////////////////////////////////////////////////////

  itk::EncapsulateMetaData(outputPtr->GetMetaDataDictionary(), MetaDataKey::TileHintX, m_OutputGridSize[0]);
  itk::EncapsulateMetaData(outputPtr->GetMetaDataDictionary(), MetaDataKey::TileHintY, m_OutputGridSize[1]);

 }

template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
 {
  Superclass::GenerateInputRequestedRegion();

  // Output requested region
  RegionType requestedRegion = this->GetOutput()->GetRequestedRegion();

  // First, align the output region
  EnlargeToAlignedRegion(requestedRegion);

  // For each image, get the requested region
  for(unsigned int i = 0; i < this->GetNumberOfInputs(); ++i)
    {
    ImageType * inputImage = static_cast<ImageType * >( Superclass::ProcessObject::GetInput(i) );

    // Compute the requested region
    RegionType inRegion;
    if (!OutputRegionToInputRegion(requestedRegion, inRegion, inputImage) )
      {
      // Image does not overlap requested region: set requested region to null
      itkDebugMacro( <<  "Image #" << i << " :\n" << inRegion << " is outside the requested region");
      inRegion.GetModifiableIndex().Fill(0);
      inRegion.GetModifiableSize().Fill(0);
      }

    // Compute the FOV-scale*FOE radius to pad
    SizeType toPad(this->GetInputReceptiveFields().at(i));
    for(unsigned int dim = 0; dim<ImageType::ImageDimension; ++dim)
      {
      int valToPad = 1 + (this->GetOutputExpressionFields().at(0)[dim] - 1) * m_OutputSpacingScale * this->GetInput(0)->GetSpacing()[dim] / this->GetInput(i)->GetSpacing()[dim] ;
      if (valToPad > toPad[dim])
        itkExceptionMacro("The input requested region of source #" << i << " is not consistent (dim "<< dim<< ")." <<
                          "Please check RF, EF, SF vs physical spacing of your image!" <<
                          "\nReceptive field: " << this->GetInputReceptiveFields().at(i)[dim] <<
                          "\nExpression field: " << this->GetOutputExpressionFields().at(0)[dim] <<
                          "\nScale factor: " << m_OutputSpacingScale <<
                          "\nReference image spacing: " << this->GetInput(0)->GetSpacing()[dim] <<
                          "\nImage " << i << " spacing: " << this->GetInput(i)->GetSpacing()[dim]);
      toPad[dim] -= valToPad;
      }

    // Pad with radius
    SmartPad(inRegion, toPad);

    // We need to avoid some extrapolation when mode is patch-based.
    // The reason is that, when some input have a lower spacing than the
    // reference image, the requested region of this lower res input image
    // can be one pixel larger when the input image regions are not physically
    // aligned.
    if (!m_FullyConvolutional)
      {
      inRegion.PadByRadius(1);
      }

    inRegion.Crop(inputImage->GetLargestPossibleRegion());

    // Update the requested region
    inputImage->SetRequestedRegion(inRegion);

    } // next image

 }

/**
 * Compute the output image
 */
template <class TInputImage, class TOutputImage>
void
TensorflowMultisourceModelFilter<TInputImage, TOutputImage>
::GenerateData()
 {
  // Output pointer and requested region
  typename TOutputImage::Pointer outputPtr = this->GetOutput();
  const RegionType outputReqRegion = outputPtr->GetRequestedRegion();

  // Get the aligned output requested region
  RegionType outputAlignedReqRegion(outputReqRegion);
  EnlargeToAlignedRegion(outputAlignedReqRegion);

  const unsigned int nInputs = this->GetNumberOfInputs();

  // Create input tensors list
  DictType inputs;


  // Populate input tensors
  for (unsigned int i = 0 ; i < nInputs ; i++)
    {
    // Input image pointer
    const ImagePointerType inputPtr = const_cast<TInputImage*>(this->GetInput(i));

    // Patch size of tensor #i
    const SizeType inputPatchSize = this->GetInputReceptiveFields().at(i);

    // Input image requested region
    const RegionType reqRegion = inputPtr->GetRequestedRegion();

    if (m_FullyConvolutional)
      {
      // Shape of input tensor #i
      tensorflow::int64 sz_n = 1;
      tensorflow::int64 sz_y = reqRegion.GetSize(1);
      tensorflow::int64 sz_x = reqRegion.GetSize(0);
      tensorflow::int64 sz_c = inputPtr->GetNumberOfComponentsPerPixel();
      tensorflow::TensorShape inputTensorShape({sz_n, sz_y, sz_x, sz_c});

      // Create the input tensor
      tensorflow::Tensor inputTensor(this->GetInputTensorsDataTypes()[i], inputTensorShape);

      // Recopy the whole input
      tf::RecopyImageRegionToTensorWithCast<TInputImage>(inputPtr, reqRegion, inputTensor, 0);

      // Input is the tensor representing the subset of image
      DictElementType input = { this->GetInputPlaceholders()[i], inputTensor };
      inputs.push_back(input);
      }
    else
      {
      // Preparing patches
      // Shape of input tensor #i
      tensorflow::int64 sz_n = outputReqRegion.GetNumberOfPixels();
      tensorflow::int64 sz_y = inputPatchSize[1];
      tensorflow::int64 sz_x = inputPatchSize[0];
      tensorflow::int64 sz_c = inputPtr->GetNumberOfComponentsPerPixel();
      tensorflow::TensorShape inputTensorShape({sz_n, sz_y, sz_x, sz_c});

      // Create the input tensor
      tensorflow::Tensor inputTensor(this->GetInputTensorsDataTypes()[i], inputTensorShape);

      // Fill the input tensor.
      // We iterate over points which are located from the index iterator
      // moving through the output image requested region
      unsigned int elemIndex = 0;
      IndexIteratorType idxIt(outputPtr, outputReqRegion);
      for (idxIt.GoToBegin(); !idxIt.IsAtEnd(); ++idxIt)
        {
        // Get the coordinates of the current output pixel
        PointType point;
        outputPtr->TransformIndexToPhysicalPoint(idxIt.GetIndex(), point);

        // Sample the i-th input patch centered on the point
        tf::SampleCenteredPatch<TInputImage>(inputPtr, point, inputPatchSize, inputTensor, elemIndex);
        elemIndex++;
        }

      // Input is the tensor of patches (aka the batch)
      DictElementType input = { this->GetInputPlaceholders()[i], inputTensor };
      inputs.push_back(input);

      } // mode is not full convolutional

    } // next input tensor

  // Run session
  // TODO: see if we print some info about inputs/outputs of the model e.g. m_OutputTensors
  TensorListType outputs;
  this->RunSession(inputs, outputs);

  // Fill the output buffer with zero value
  outputPtr->SetBufferedRegion(outputReqRegion);
  outputPtr->Allocate();
  outputPtr->FillBuffer(m_NullPixel);

  // Get output tensors
  int bandOffset = 0;
  for (unsigned int i = 0 ; i < outputs.size() ; i++)
    {
    // The offset (i.e. the starting index of the channel for the output tensor) is updated
    // during this call
    // TODO: implement a generic strategy enabling expression field copy in patch-based mode (see tf::CopyTensorToImageRegion)
    try
      {
      tf::CopyTensorToImageRegion<TOutputImage> (outputs[i],
          outputAlignedReqRegion, outputPtr, outputReqRegion, bandOffset);
      }
    catch( itk::ExceptionObject & err )
      {
      std::stringstream debugMsg = this->GenerateDebugReport(inputs);
      itkExceptionMacro("Error occurred during tensor to image conversion.\n"
          << "Context: " << debugMsg.str()
          << "Error:" << err);
      }
    }

 }


} // end namespace otb


#endif
