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

// Filter
#include "otbTensorflowSampler.h"

// Stack
#include "otbTensorflowSource.h"

namespace otb
{

namespace Wrapper
{

class PatchesExtraction : public Application
{
public:
  /** Standard class typedefs. */
  typedef PatchesExtraction                   Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(PatchesExtraction, Application);

  /** Filter typedef */
  typedef otb::TensorflowSampler<FloatVectorImageType, VectorDataType> SamplerType;

  /** Typedefs for image concatenation */
  typedef TensorflowSource<FloatVectorImageType>                       TFSourceType;

  //
  // Store stuff related to one source
  //
  struct SourceBundle
  {
    TFSourceType                       m_ImageSource;   // Image source
    FloatVectorImageType::SizeType     m_PatchSize;          // Patch size

    unsigned int                       m_NumberOfElements;  // Number of output samples

    std::string                        m_KeyIn;   // Key of input image list
    std::string                        m_KeyOut;  // Key of output samples image
    std::string                        m_KeyPszX; // Key for samples sizes X
    std::string                        m_KeyPszY; // Key for samples sizes Y
  };


  //
  // Add an input source, which includes:
  // -an input image list
  // -an output image (samples)
  // -an input patchsize (dimensions of samples)
  //
  void AddAnInputImage()
  {
    // Number of source
    unsigned int inputNumber = m_Bundles.size() + 1;

    // Create keys and descriptions
    std::stringstream ss_group_key, ss_desc_group, ss_key_in, ss_key_out, ss_desc_in,
    ss_desc_out, ss_key_dims_x, ss_desc_dims_x, ss_key_dims_y, ss_desc_dims_y;
    ss_group_key   << "source"                    << inputNumber;
    ss_desc_group  << "Parameters for source "    << inputNumber;
    ss_key_out     << ss_group_key.str()          << ".out";
    ss_desc_out    << "Output patches for image " << inputNumber;
    ss_key_in      << ss_group_key.str()          << ".il";
    ss_desc_in     << "Input image(s) "           << inputNumber;
    ss_key_dims_x  << ss_group_key.str()          << ".patchsizex";
    ss_desc_dims_x << "X patch size for image "   << inputNumber;
    ss_key_dims_y  << ss_group_key.str()          << ".patchsizey";
    ss_desc_dims_y << "Y patch size for image "   << inputNumber;

    // Populate group
    AddParameter(ParameterType_Group,          ss_group_key.str(),  ss_desc_group.str());
    AddParameter(ParameterType_InputImageList, ss_key_in.str(),     ss_desc_in.str() );
    AddParameter(ParameterType_OutputImage,    ss_key_out.str(),    ss_desc_out.str());
    AddParameter(ParameterType_Int,            ss_key_dims_x.str(), ss_desc_dims_x.str());
    SetMinimumParameterIntValue               (ss_key_dims_x.str(), 1);
    AddParameter(ParameterType_Int,            ss_key_dims_y.str(), ss_desc_dims_y.str());
    SetMinimumParameterIntValue               (ss_key_dims_y.str(), 1);

    // Add a new bundle
    SourceBundle bundle;
    bundle.m_KeyIn   = ss_key_in.str();
    bundle.m_KeyOut  = ss_key_out.str();
    bundle.m_KeyPszX = ss_key_dims_x.str();
    bundle.m_KeyPszY = ss_key_dims_y.str();

    m_Bundles.push_back(bundle);

  }

  //
  // Prepare bundles from the number of points
  //
  void PrepareInputs()
  {
    for (auto& bundle: m_Bundles)
    {
      // Create a stack of input images
      FloatVectorImageListType::Pointer list = GetParameterImageList(bundle.m_KeyIn);
      bundle.m_ImageSource.Set(list);

      // Patch size
      bundle.m_PatchSize[0] = GetParameterInt(bundle.m_KeyPszX);
      bundle.m_PatchSize[1] = GetParameterInt(bundle.m_KeyPszY);
    }
  }

  void DoUpdateParameters()
  {
  }

  void DoInit()
  {

    // Documentation
    SetName("PatchesExtraction");
    SetDescription("This application extracts patches in multiple input images. Change "
        "the " + tf::ENV_VAR_NAME_NSOURCES + " environment variable to set the number of "
        "sources.");
    SetDocLongDescription("The application takes an input vector layer which is a set of "
        "points, typically the output of the \"SampleSelection\" or the \"LabelImageSampleSelection\" "
        "application to sample patches in the input images (samples are centered on the points). "
        "A \"source\" parameters group is composed of (i) an input image list (can be "
        "one image e.g. high res. image, or multiple e.g. time series), (ii) the size "
        "of the patches to sample, and (iii) the output images of patches which will "
        "be generated at the end of the process. The example below show how to "
        "set the samples sizes. For a SPOT6 image for instance, the patch size can "
        "be 64x64 and for an input Sentinel-2 time series the patch size could be "
        "1x1. Note that if a dimension size is not defined, the largest one will "
        "be used (i.e. input image dimensions. The number of input sources can be changed "
        "at runtime by setting the system environment variable " + tf::ENV_VAR_NAME_NSOURCES);

    SetDocAuthors("Remi Cresson");

    AddDocTag(Tags::Learning);

    // Input/output images
    AddAnInputImage();
    for (int i = 1; i < tf::GetNumberOfSources() ; i++)
      AddAnInputImage();

    // Input vector data
    AddParameter(ParameterType_InputVectorData, "vec", "Positions of the samples (must be in the same projection as input image)");

    // No data parameters
    AddParameter(ParameterType_Bool, "usenodata", "Reject samples that have no-data value");
    MandatoryOff                    ("usenodata");
    AddParameter(ParameterType_Float, "nodataval", "No data value (used only if \"usenodata\" is on)");
    SetDefaultParameterFloat(         "nodataval", 0.0);

    // Output label
    AddParameter(ParameterType_OutputImage, "outlabels", "output labels");
    SetDefaultOutputPixelType              ("outlabels", ImagePixelType_uint8);
    MandatoryOff                           ("outlabels");

    // Class field
    AddParameter(ParameterType_String, "field", "field of class in the vector data");

    // Examples values
    SetDocExampleParameterValue("vec",                "points.sqlite");
    SetDocExampleParameterValue("source1.il",         "$s2_list");
    SetDocExampleParameterValue("source1.patchsizex", "16");
    SetDocExampleParameterValue("source1.patchsizey", "16");
    SetDocExampleParameterValue("field",              "class");
    SetDocExampleParameterValue("source1.out",        "outpatches_16x16.tif");
    SetDocExampleParameterValue("outlabels",          "outlabels.tif");

  }

  void DoExecute()
  {

    PrepareInputs();

    // Setup the filter
    SamplerType::Pointer sampler = SamplerType::New();
    sampler->SetInputVectorData(GetParameterVectorData("vec"));
    sampler->SetField(GetParameterAsString("field"));
    sampler->SetRejectPatchesWithNodata(GetParameterInt("usenodata")==1);
    sampler->SetNodataValue(GetParameterFloat("nodataval"));
    for (auto& bundle: m_Bundles)
    {
      sampler->PushBackInputWithPatchSize(bundle.m_ImageSource.Get(), bundle.m_PatchSize);
    }

    // Run the filter
    AddProcess(sampler, "Sampling patches");
    sampler->Update();

    // Show numbers
    otbAppLogINFO("Number of samples collected: " << sampler->GetNumberOfAcceptedSamples());
    otbAppLogINFO("Number of samples rejected : " << sampler->GetNumberOfRejectedSamples());

    // Save patches image
    for (unsigned int i = 0 ; i < m_Bundles.size() ; i++)
    {
      SetParameterOutputImage(m_Bundles[i].m_KeyOut, sampler->GetOutputPatchImages()[i]);
    }


    // Save label image (if needed)
    if (HasValue("outlabels"))
    {
      SetParameterOutputImage("outlabels", sampler->GetOutputLabelImage());
    }

  }
private:
  std::vector<SourceBundle> m_Bundles;

}; // end of class

} // end namespace wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::PatchesExtraction )
