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

    // Input image
    AddParameter(ParameterType_InputImage, "in", "input patches image");

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
    AddChoice("op.merge", "Merge two patches images into one");
    AddParameter(ParameterType_InputImage, "op.merge.in", "patches image to merge");

    // Output
    AddParameter(ParameterType_OutputImage, "out", "Output patches image");

    AddRAMParameter();

  }

  void CheckPatchesDimensions(std::string key)
  {
    FloatVectorImageType::Pointer in1 = GetParameterFloatVectorImage("in");
    FloatVectorImageType::Pointer in2 = GetParameterFloatVectorImage(key);

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
    std::string key = "op.merge.in";

    // Check patches consistency
    CheckPatchesDimensions(key);

    // Get images pointers
    FloatVectorImageType::Pointer in = GetParameterFloatVectorImage("in");
    FloatVectorImageType::Pointer in2 = GetParameterFloatVectorImage(key);

    FloatVectorImageType::RegionType in1Region = in->GetLargestPossibleRegion();
    FloatVectorImageType::RegionType in2Region = in2->GetLargestPossibleRegion();

    // Allocate output image
    FloatVectorImageType::RegionType outRegion;
    outRegion.GetModifiableIndex().Fill(0);
    outRegion.GetModifiableSize()[0] = GetParameterInt("patches.sizex");
    outRegion.GetModifiableSize()[1] = in1Region.GetSize(1) + in2Region.GetSize(1);
    m_Out = FloatVectorImageType::New();
    m_Out->SetRegions(outRegion);
    m_Out->SetNumberOfComponentsPerPixel(in->GetNumberOfComponentsPerPixel());
    m_Out->Allocate();

    // Read input images
    otbAppLogINFO("Reading input images...")
    tf::PropagateRequestedRegion<FloatVectorImageType>(in, in1Region);
    tf::PropagateRequestedRegion<FloatVectorImageType>(in2, in2Region);

    // Recopy
    otbAppLogINFO("Merging...")
    itk::ImageRegionIterator<FloatVectorImageType> outIt(m_Out, outRegion);
    itk::ImageRegionConstIterator<FloatVectorImageType> inIt(in, in1Region);
    for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt, ++outIt)
      outIt.Set(inIt.Get());
    itk::ImageRegionConstIterator<FloatVectorImageType> inIt2(in2, in2Region);
    for (inIt2.GoToBegin(); !inIt2.IsAtEnd(); ++inIt2, ++outIt)
      outIt.Set(inIt2.Get());

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
