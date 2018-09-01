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

class TrainClassifierFromDeepFeatures : public CompositeApplication
{
public:
  /** Standard class typedefs. */
  typedef TrainClassifierFromDeepFeatures              Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(TrainClassifierFromDeepFeatures, otb::Wrapper::CompositeApplication);

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

  SetName("TrainClassifierFromDeepFeatures");
  SetDescription("Train a classifier from deep net based features of an image and training vector data.");

  // Documentation
  SetDocName("TrainClassifierFromDeepFeatures");
  SetDocLongDescription("See TrainImagesClassifier application");
  SetDocLimitations("None");
  SetDocAuthors("Remi Cresson");
  SetDocSeeAlso(" ");

  ClearApplications();

  // Add applications
  AddApplication("TrainImagesClassifier",  "train"  , "Train images classifier" );
  AddApplication("TensorflowModelServe" ,  "tfmodel", "Serve the TF model"      );

  // Model shared parameters
  AddAnInputImage();
  for (int i = 1; i < tf::GetNumberOfSources() ; i++)
  {
    AddAnInputImage(i);
  }
  ShareParameter("model", "tfmodel.model", "Deep net model parameters", "Deep net model parameters");
  ShareParameter("output", "tfmodel.output", "Deep net outputs parameters", "Deep net outputs parameters");
  ShareParameter("optim", "tfmodel.optim", "This group of parameters allows optimization of processing time", "This group of parameters allows optimization of processing time");

  // Train shared parameters
  ShareParameter("vd"  , "train.io.vd"      , "Input vector data list"      , "Input vector data list" );
  ShareParameter("valid" , "train.io.valid"        , "Validation vector data list" , "Validation vector data list"          );
  ShareParameter("out"    , "train.io.out" , "Output model" , "Output model"   );
  ShareParameter("confmatout"    , "train.io.confmatout" , "Output model confusion matrix"  , "Output model confusion matrix"    );

  // Shared parameter groups
  ShareParameter("sample"    , "train.sample"       , "Training and validation samples parameters" , "Training and validation samples parameters" );
  ShareParameter("elev"    , "train.elev"       , "Elevation management" , "Elevation management" );
  ShareParameter("classifier"    , "train.classifier"       , "Classifier" , "Classifier" );
  ShareParameter("rand"    , "train.rand"       , "User defined rand seed" , "User defined rand seed" );

  }


  void DoUpdateParameters()
  {
    UpdateInternalParameters("train");
  }

  void DoExecute()
  {
    ExecuteInternal("tfmodel");
    GetInternalApplication("train")->AddImageToParameterInputImageList("io.il", GetInternalApplication("tfmodel")->GetParameterOutputImage("out"));
    UpdateInternalParameters("train");
    ExecuteInternal("train");
  }   // DOExecute()

  void AfterExecuteAndWriteOutputs()
  {
    // Nothing to do
  }

};
}
}

OTB_APPLICATION_EXPORT( otb::Wrapper::TrainClassifierFromDeepFeatures )
