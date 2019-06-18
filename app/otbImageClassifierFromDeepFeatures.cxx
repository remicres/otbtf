/*=========================================================================

  Copyright (c) Remi Cresson (IRSTEA). All rights reserved.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "itkFixedArray.h"
#include "itkObjectFactory.h"

// Elevation handler
#include "otbWrapperElevationParametersHandler.h"
#include "otbWrapperApplicationFactory.h"
#include "otbWrapperCompositeApplication.h"

// Application engine
#include "otbStandardFilterWatcher.h"
#include "itkFixedArray.h"

// TF (used to get the environment variable for the number of inputs)
#include "otbTensorflowCommon.h"

namespace otb
{

namespace Wrapper
{

class ImageClassifierFromDeepFeatures : public CompositeApplication
{
public:
  /** Standard class typedefs. */
  typedef ImageClassifierFromDeepFeatures              Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(ImageClassifierFromDeepFeatures, otb::Wrapper::CompositeApplication);

private:

  //
  // Add an input source, which includes:
  // -an input image list
  // -an input patchsize (dimensions of samples)
  //
  void AddAnInputImage(int inputNumber = 0)
  {
    inputNumber++;

    // Create keys and descriptions
    std::stringstream ss_key_group, ss_desc_group;
    ss_key_group << "source" << inputNumber;
    ss_desc_group << "Parameters for source " << inputNumber;

    // Populate group
    ShareParameter(ss_key_group.str(), "tfmodel." + ss_key_group.str(), ss_desc_group.str());

  }


  void DoInit()
  {

    SetName("ImageClassifierFromDeepFeatures");
    SetDescription("Classify image using features from a deep net and an OTB machine learning classification model");

    // Documentation
    SetDocLongDescription("See ImageClassifier application");
    SetDocLimitations("None");
    SetDocAuthors("Remi Cresson");
    SetDocSeeAlso(" ");

    AddDocTag(Tags::Learning);

    ClearApplications();

    // Add applications
    AddApplication("ImageClassifier",      "classif", "Images classifier"  );
    AddApplication("TensorflowModelServe", "tfmodel", "Serve the TF model" );

    // Model shared parameters
    AddAnInputImage();
    for (int i = 1; i < tf::GetNumberOfSources() ; i++)
    {
      AddAnInputImage(i);
    }
    ShareParameter("deepmodel",  "tfmodel.model",
        "Deep net model parameters",      "Deep net model parameters");
    ShareParameter("output",     "tfmodel.output",
        "Deep net outputs parameters",
        "Deep net outputs parameters");
    ShareParameter("optim", "tfmodel.optim",
        "This group of parameters allows optimization of processing time",
        "This group of parameters allows optimization of processing time");

    // Classify shared parameters
    ShareParameter("model"      , "classif.model"      , "Model file"          , "Model file"          );
    ShareParameter("imstat"     , "classif.imstat"     , "Statistics file"     , "Statistics file"     );
    ShareParameter("nodatalabel", "classif.nodatalabel", "Label mask value"    , "Label mask value"    );
    ShareParameter("out"        , "classif.out"        , "Output image"        , "Output image"        );
    ShareParameter("confmap"    , "classif.confmap"    , "Confidence map image", "Confidence map image");
    ShareParameter("ram"        , "classif.ram"        , "Ram"                 , "Ram"                 );

  }


  void DoUpdateParameters()
  {
    UpdateInternalParameters("classif");
  }

  void DoExecute()
  {
    ExecuteInternal("tfmodel");
    GetInternalApplication("classif")->SetParameterInputImage("in", GetInternalApplication("tfmodel")->GetParameterOutputImage("out"));
    UpdateInternalParameters("classif");
    ExecuteInternal("classif");
  }   // DOExecute()

  void AfterExecuteAndWriteOutputs()
  {
    // Nothing to do
  }
};
} // namespace Wrapper
} // namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::ImageClassifierFromDeepFeatures )
