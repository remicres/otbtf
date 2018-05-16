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

#include "vnl/vnl_vector.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

// image utils
#include "otbTensorflowCommon.h"

namespace otb
{

namespace Wrapper
{

class LabelImageSampleSelection : public Application
{
public:
  /** Standard class typedefs. */
  typedef LabelImageSampleSelection          Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(LabelImageSampleSelection, Application);

  /** Vector data typedefs */
  typedef VectorDataType::DataTreeType                 DataTreeType;
  typedef itk::PreOrderTreeIterator<DataTreeType>      TreeIteratorType;
  typedef VectorDataType::DataNodeType                 DataNodeType;
  typedef DataNodeType::Pointer                        DataNodePointer;

  /** typedefs */
  typedef Int16ImageType                               LabelImageType;
  typedef unsigned int                                 IndexValueType;

  void DoUpdateParameters()
  {
  }

  /*
   * Display the percentage
   */
  void ShowProgress(unsigned int count, unsigned int total, unsigned int step = 1000)
  {
    if (count % step == 0)
    {
      std::cout << std::setprecision(3) << "\r" << (100.0 * count / (float) total) << "%      " << std::flush;
    }
  }

  void ShowProgressDone()
  {
    std::cout << "\rDone      " << std::flush;
    std::cout << std::endl;
  }

  void DoInit()
  {

    // Documentation
    SetName("LabelImageSampleSelection");
    SetDocName("LabelImageSampleSelection");
    SetDescription("This application extracts points from an input label image. "
        "This application is like \"SampleSelection\", but uses an input label "
        "image, rather than an input vector data.");
    SetDocLongDescription("This application produces a vector data containing "
        "a set of points centered on the pixels of the input label image. "
        "The user can control the number of points. The default strategy consists "
        "in producing the same number of points in each class. If one class has a "
        "smaller number of points than requested, this one is adjusted.");

    SetDocAuthors("Remi Cresson");

    // Input terrain truth
    AddParameter(ParameterType_InputImage, "inref", "input terrain truth");

    // Strategy
    AddParameter(ParameterType_Choice, "strategy", "Sampling strategy");

    AddChoice("strategy.constant","Set the same samples counts for all classes");
    SetParameterDescription("strategy.constant","Set the same samples counts for all classes");

    AddParameter(ParameterType_Int, "strategy.constant.nb", "Number of samples for all classes");
    SetParameterDescription("strategy.constant.nb", "Number of samples for all classes");
    SetMinimumParameterIntValue("strategy.constant.nb",1);
    SetDefaultParameterInt("strategy.constant.nb",1000);

    AddChoice("strategy.total","Set the total number of samples to generate, and use class proportions.");
    SetParameterDescription("strategy.total","Set the total number of samples to generate, and use class proportions.");
    AddParameter(ParameterType_Int,"strategy.total.v","The number of samples to generate");
    SetParameterDescription("strategy.total.v","The number of samples to generate");
    SetMinimumParameterIntValue("strategy.total.v",1);
    SetDefaultParameterInt("strategy.total.v",1000);

    AddChoice("strategy.smallest","Set same number of samples for all classes, with the smallest class fully sampled");
    SetParameterDescription("strategy.smallest","Set same number of samples for all classes, with the smallest class fully sampled");

    AddChoice("strategy.all","Take all samples");
    SetParameterDescription("strategy.all","Take all samples");

    // Default strategy : smallest
    SetParameterString("strategy","constant");

    // Input no-data value
    AddParameter(ParameterType_Int, "nodata", "nodata value");
    MandatoryOn                    ("nodata");
    SetDefaultParameterInt         ("nodata", -1);

    // Padding
    AddParameter(ParameterType_Int, "pad", "padding, in pixels");
    SetDefaultParameterInt         ("pad", 0);
    MandatoryOff                   ("pad");

    // Output points
    AddParameter(ParameterType_OutputVectorData, "outvec", "output set of points");

    // Some example
    SetDocExampleParameterValue("inref", "rasterized_terrain_truth.tif");
    SetDocExampleParameterValue("outvec", "terrain_truth_points_sel.sqlite");

    AddRAMParameter();

  }


  void DoExecute()
  {

    // Count the number of pixels in each class
    const LabelImageType::InternalPixelType MAX_NB_OF_CLASSES =
        itk::NumericTraits<LabelImageType::InternalPixelType>::max();;
    LabelImageType::InternalPixelType class_begin = MAX_NB_OF_CLASSES;
    LabelImageType::InternalPixelType class_end = 0;
    vnl_vector<IndexValueType> tmp_number_of_samples(MAX_NB_OF_CLASSES, 0);

    otbAppLogINFO("Computing number of pixels in each class");

    // Explicit streaming over the input target image, based on the RAM parameter
    typedef otb::RAMDrivenStrippedStreamingManager<FloatVectorImageType> StreamingManagerType;
    StreamingManagerType::Pointer m_StreamingManager = StreamingManagerType::New();
    m_StreamingManager->SetAvailableRAMInMB(GetParameterInt("ram"));

    // We pad the image, if this is requested by the user
    LabelImageType::Pointer inputImage = GetParameterInt16Image("inref");
    LabelImageType::RegionType entireRegion = inputImage->GetLargestPossibleRegion();
    entireRegion.ShrinkByRadius(GetParameterInt("pad"));
    m_StreamingManager->PrepareStreaming(inputImage, entireRegion );

    // Get nodata value
    const LabelImageType::InternalPixelType nodata = GetParameterInt("nodata");

    // First iteration to count the objects in each class
    int m_NumberOfDivisions = m_StreamingManager->GetNumberOfSplits();
    for (int m_CurrentDivision = 0; m_CurrentDivision < m_NumberOfDivisions; m_CurrentDivision++)
    {
      LabelImageType::RegionType streamRegion = m_StreamingManager->GetSplit(m_CurrentDivision);
      tf::PropagateRequestedRegion<LabelImageType>(inputImage, streamRegion);
      itk::ImageRegionConstIterator<LabelImageType> inIt (inputImage, streamRegion);
      for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt)
      {
        LabelImageType::InternalPixelType pixVal = inIt.Get();
        if (pixVal != nodata)
        {
          // Update min and max value
          if (pixVal > class_end)
            class_end = pixVal;
          if (pixVal < class_begin)
            class_begin = pixVal;

          tmp_number_of_samples(pixVal)++;
        }
      }

      ShowProgress(m_CurrentDivision, m_NumberOfDivisions, 1);
    }
    ShowProgressDone();

    // Number of classes
    const LabelImageType::InternalPixelType number_of_classes = class_end - class_begin + 1;

    // Number of samples in each class (counted)
    vnl_vector<IndexValueType> number_of_samples = tmp_number_of_samples.extract(number_of_classes, class_begin);

    // Number of samples in each class (target)
    vnl_vector<IndexValueType> target_number_of_samples(number_of_classes, 0);

    otbAppLogINFO( "Number of classes: " << number_of_classes <<
        " starting from " << class_begin <<
        " to " << class_end << " (no-data is " << nodata << ")");
    otbAppLogINFO( "Number of pixels in each class: " << number_of_samples );

    // Check the smallest number of samples amongst classes
    IndexValueType min_elem_in_class = itk::NumericTraits<IndexValueType>::max();
    for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
      min_elem_in_class = vcl_min(min_elem_in_class, number_of_samples[classIdx]);

    // If one class is empty, throw an error
    if (min_elem_in_class == 0)
    {
      otbAppLogFATAL("There is at least one class with no sample!")
    }

    // Sampling step for each classes
    vnl_vector<IndexValueType> step_for_class(number_of_classes, 0);

    // Compute the sampling step for each classes, depending on the chosen strategy
    switch (this->GetParameterInt("strategy"))
    {
    // constant
    case 0:
    {
      // Set the target number of samples in each class
      target_number_of_samples.fill(GetParameterInt("strategy.constant.nb"));

      // re adjust the number of samples to select in each class
      if (min_elem_in_class < target_number_of_samples[0])
      {
        otbAppLogWARNING("Smallest class has " << min_elem_in_class <<
            " samples but a number of " << target_number_of_samples[0] <<
            " is given. Using " << min_elem_in_class);
        target_number_of_samples.fill( min_elem_in_class );
      }

      // Compute the sampling step
      for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
        step_for_class[classIdx] = number_of_samples[classIdx] / target_number_of_samples[classIdx];
    }
    break;

    // total
    case 1:
    {
      // Compute the sampling step
      IndexValueType step = number_of_samples.sum() / this->GetParameterInt("strategy.total.v");
      if (step == 0)
      {
        otbAppLogWARNING("The number of samples available is smaller than the required number of samples. " <<
            "Setting sampling step to 1.");
        step = 1;
      }
      step_for_class.fill(step);

      // Compute the target number of samples
      for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
        target_number_of_samples[classIdx] = number_of_samples[classIdx] / step;

    }
    break;

    // smallest
    case 2:
    {
      // Set the target number of samples to the smallest class
      target_number_of_samples.fill( min_elem_in_class );

      // Compute the sampling step
      for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
        step_for_class[classIdx] = number_of_samples[classIdx] / target_number_of_samples[classIdx];

    }
    break;

    // All
    case 3:
    {
      // Easy
      step_for_class.fill(1);
      target_number_of_samples = number_of_samples;
    }
    break;
    default:
      otbAppLogFATAL("Strategy mode unknown :"<<this->GetParameterString("strategy"));
      break;
    }

    // Print quick summary
    otbAppLogINFO("Sampling summary:");
    otbAppLogINFO("\tClass\tStep\tTot");
    for (LabelImageType::InternalPixelType i = 0 ; i < number_of_classes ; i++)
    {
      vnl_vector<int> tmp (3,0);
      tmp[0] = i + class_begin;
      tmp[1] = step_for_class[i];
      tmp[2] = target_number_of_samples[i];
      otbAppLogINFO("\t" << tmp);
    }

    // Create a new vector data
    // TODO: how to pre-allocate the datatree?
    m_OutVectorData = VectorDataType::New();
    DataTreeType::Pointer tree = m_OutVectorData->GetDataTree();
    DataNodePointer root = tree->GetRoot()->Get();
    DataNodePointer document = DataNodeType::New();
    document->SetNodeType(DOCUMENT);
    tree->Add(document, root);

    // Duno if this makes sense?
    m_OutVectorData->SetProjectionRef(inputImage->GetProjectionRef());
    m_OutVectorData->SetOrigin(inputImage->GetOrigin());
    m_OutVectorData->SetSpacing(inputImage->GetSpacing());

    // Second iteration, to prepare the samples
    vnl_vector<IndexValueType> sampledCount(number_of_classes, 0);
    vnl_vector<IndexValueType> iteratorCount(number_of_classes, 0);
    IndexValueType n_tot = 0;
    const IndexValueType target_n_tot = target_number_of_samples.sum();
    for (int m_CurrentDivision = 0; m_CurrentDivision < m_NumberOfDivisions; m_CurrentDivision++)
    {
      LabelImageType::RegionType streamRegion = m_StreamingManager->GetSplit(m_CurrentDivision);
      tf::PropagateRequestedRegion<LabelImageType>(inputImage, streamRegion);
      itk::ImageRegionConstIterator<LabelImageType> inIt (inputImage, streamRegion);

      for (inIt.GoToBegin() ; !inIt.IsAtEnd() ; ++inIt)
      {
        LabelImageType::InternalPixelType classVal = inIt.Get();

        if (classVal != nodata)
        {
          classVal -= class_begin;

          // Update the current position
          iteratorCount[classVal]++;

          // Every Xi samples (Xi is the step for class i)
          if (iteratorCount[classVal] % ((int) step_for_class[classVal]) == 0 &&
              sampledCount[classVal] < target_number_of_samples[classVal])
          {
            // Add this sample
            sampledCount[classVal]++;
            n_tot++;
            ShowProgress(n_tot, target_n_tot);

            // Create a point
            LabelImageType::PointType geo;
            inputImage->TransformIndexToPhysicalPoint(inIt.GetIndex(), geo);
            DataNodeType::PointType point;
            point[0] = geo[0];
            point[1] = geo[1];

            // Add point to the VectorData tree
            DataNodePointer newDataNode = DataNodeType::New();
            newDataNode->SetPoint(point);
            newDataNode->SetFieldAsInt("class", static_cast<int>(classVal));
            tree->Add(newDataNode, document);

          } // sample this one
        }
      } // next pixel
    } // next streaming region
    ShowProgressDone();

    otbAppLogINFO( "Number of samples in each class: " << sampledCount );

    otbAppLogINFO( "Writing output vector data");

    SetParameterOutputVectorData("outvec", m_OutVectorData);

  }

private:
  VectorDataType::Pointer m_OutVectorData;

}; // end of class

} // end namespace wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::LabelImageSampleSelection )
