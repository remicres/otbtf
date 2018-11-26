/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "itkFixedArray.h"
#include "itkObjectFactory.h"
#include "otbWrapperApplicationFactory.h"

// Application engine
#include "otbStandardFilterWatcher.h"
#include "itkFixedArray.h"

// Image
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

// image utils
#include "otbTensorflowCommon.h"

namespace otb
{

namespace Wrapper
{

class PatchesManipulation : public Application
{
public:
  /** Standard class typedefs. */
  typedef PatchesManipulation          Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(PatchesManipulation, Application);

  typedef otb::RAMDrivenStrippedStreamingManager<FloatVectorImageType> StreamingManagerType;

  void DoUpdateParameters()
  {
  }


  void DoInit()
  {

    // Documentation
    SetName("PatchesManipulation");
    SetDocName("PatchesManipulation");
    SetDescription("This application enables to edit patches images.");
    SetDocLongDescription("This application provide various operations for patches "
        "images edition. ");

    SetDocAuthors("Remi Cresson");

    // Patches size
    AddParameter(ParameterType_Group, "patches", "grid settings");
    AddParameter(ParameterType_Int, "patches.sizex", "Patch size X");
    SetMinimumParameterIntValue    ("patches.sizex", 1);
    MandatoryOn                    ("patches.sizex");
    AddParameter(ParameterType_Int, "patches.sizey", "Patch size Y");
    SetMinimumParameterIntValue    ("patches.sizey", 1);
    MandatoryOn                    ("patches.sizey");

    // Operation
    AddParameter(ParameterType_Choice, "op", "Operation");
    AddChoice("op.merge", "Merge multiple patches images into one");
    AddParameter(ParameterType_InputImageList, "op.merge.il", "patches images to merge");

    // Output
    AddParameter(ParameterType_OutputImage, "out", "Output patches image");

    AddRAMParameter();

  }

  void CheckPatchesDimensions(FloatVectorImageType::Pointer in1, FloatVectorImageType::Pointer in2)
  {
    FloatVectorImageType::SizeType size1 = in1->GetLargestPossibleRegion().GetSize();
    FloatVectorImageType::SizeType size2 = in2->GetLargestPossibleRegion().GetSize();
    unsigned int nbands1 = in1->GetNumberOfComponentsPerPixel();
    unsigned int nbands2 = in2->GetNumberOfComponentsPerPixel();

    if (nbands1 != nbands2)
      otbAppLogFATAL("Patches must have the same number of channels");

    if (static_cast<int>(size1[0]) != GetParameterInt("patches.sizex"))
      otbAppLogFATAL("Input patches image width not consistent with patch size x");

    if (size1[1] % GetParameterInt("patches.sizey") != 0)
      otbAppLogFATAL("Input patches image height is " << size1[1] << " which is not a multiple of " << GetParameterInt("patches.sizey"));

    if (size2[1] % GetParameterInt("patches.sizey") != 0)
      otbAppLogFATAL("Patches image height is " << size2[1] << " which is not a multiple of " << GetParameterInt("patches.sizey"));

    if (size2[0] != size1[0])
      otbAppLogFATAL("Input patches images must have the same width!");

    unsigned int pszy1 = size1[1] / GetParameterInt("patches.sizey");
    unsigned int pszy2 = size2[1] / GetParameterInt("patches.sizey");

    if (pszy1 != pszy2)
      otbAppLogFATAL("Patches must have the same height!");

  }

  /*
   * Merge two patches images into one
   * TODO:
   * Use ImageToImage to create a filter that do this in a streamable way
   */
  void MergePatches()
  {
    std::string key = "op.merge.il";

    FloatVectorImageListType::Pointer imagesList = this->GetParameterImageList(key);
    unsigned int nImgs = imagesList->Size();

    otbAppLogINFO("Number of patches images: " << nImgs);

    // Check patches consistency and count rows
    FloatVectorImageType::IndexValueType nrows = imagesList->GetNthElement(0)->GetLargestPossibleRegion().GetSize(1);
    FloatVectorImageType::Pointer img0 = imagesList->GetNthElement(0);
    for (unsigned int i = 1; i < nImgs ; i++)
    {
      FloatVectorImageType::Pointer img = imagesList->GetNthElement(i);
      CheckPatchesDimensions(img0, img);
      nrows += img->GetLargestPossibleRegion().GetSize(1);
    }

    // Allocate output image
    FloatVectorImageType::RegionType outRegion;
    outRegion.GetModifiableIndex().Fill(0);
    outRegion.GetModifiableSize()[0] = GetParameterInt("patches.sizex");
    outRegion.GetModifiableSize()[1] = nrows;
    m_Out = FloatVectorImageType::New();
    m_Out->SetRegions(outRegion);
    m_Out->SetNumberOfComponentsPerPixel(img0->GetNumberOfComponentsPerPixel());

    otbAppLogINFO("Allocating output image of " << outRegion.GetSize() <<
        " pixels with " << img0->GetNumberOfComponentsPerPixel() << " channels");

    m_Out->Allocate();

    // Read input images
    itk::ImageRegionIterator<FloatVectorImageType> outIt(m_Out, outRegion);
    outIt.GoToBegin();
    for (unsigned int i = 0; i < nImgs ; i++)
    {
      // Get current image
      FloatVectorImageType::Pointer img = imagesList->GetNthElement(i);
      FloatVectorImageType::RegionType region = img->GetLargestPossibleRegion();
      otbAppLogINFO("Processing input image " << (i+1) << "/" << nImgs);

      // Recopy
      tf::PropagateRequestedRegion<FloatVectorImageType>(img, region);
      itk::ImageRegionConstIterator<FloatVectorImageType> inIt(img, region);
      for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++outIt)
        outIt.Set(inIt.Get());

      // Release data bulk
      img->PrepareForNewData();
    }

    SetParameterOutputImage("out", m_Out);
  }

  void DoExecute()
  {

    if (GetParameterAsString("op").compare("merge") == 0)
      {
      otbAppLogINFO("Operation is merge");
      MergePatches();
      }
    else
      otbAppLogFATAL("Please select an existing operation");

  }

private:
  FloatVectorImageType::Pointer m_Out;


}; // end of class

} // end namespace wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::PatchesManipulation )
