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
#include "itkUnaryFunctorImageFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkBinaryErodeImageFilter.h"
#include "otbStreamingResampleImageFilter.h"

// image utils
#include "otbTensorflowCommon.h"

// Functor to retrieve nodata
template<class TPixel, class OutputPixel>
class IsNoData
{
public:
  IsNoData(){}
  ~IsNoData(){}

  inline OutputPixel operator()( const TPixel & A ) const
  {
    for (unsigned int band = 0 ; band < A.Size() ; band++)
      {
      if (A[band] != m_NoDataValue)
        return 1;
      }
    return 0;
  }

  void SetNoDataValue(typename TPixel::ValueType value)
  {
    m_NoDataValue = value;
  }

private:
  typename TPixel::ValueType m_NoDataValue;
};

namespace otb
{

namespace Wrapper
{

class PatchesSelection : public Application
{
public:
  /** Standard class typedefs. */
  typedef PatchesSelection          Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(PatchesSelection, Application);

  /** Vector data typedefs */
  typedef VectorDataType::DataTreeType                 DataTreeType;
  typedef itk::PreOrderTreeIterator<DataTreeType>      TreeIteratorType;
  typedef VectorDataType::DataNodeType                 DataNodeType;
  typedef DataNodeType::Pointer                        DataNodePointer;

  /** typedefs */
  typedef IsNoData<FloatVectorImageType::PixelType, UInt8ImageType::PixelType > IsNoDataFunctorType;
  typedef itk::UnaryFunctorImageFilter<FloatVectorImageType, UInt8ImageType, IsNoDataFunctorType> IsNoDataFilterType;

  typedef itk::FlatStructuringElement<2>                                         StructuringType;
  typedef StructuringType::RadiusType                                            RadiusType;

  typedef itk::BinaryErodeImageFilter<UInt8ImageType, UInt8ImageType, StructuringType> MorphoFilterType;

  typedef otb::StreamingResampleImageFilter<UInt8ImageType,UInt8ImageType> PadFilterType;

  void DoUpdateParameters()
  {
  }


  void DoInit()
  {

    // Documentation
    SetName("PatchesSelection");
    SetDocName("PatchesSelection");
    SetDescription("This application generate points sampled at regular interval over "
        "the input image region. The grid size and spacing can be configured.");
    SetDocLongDescription("This application produces a vector data containing "
        "a set of points centered on the patches lying in the valid regions of the input image. ");

    SetDocAuthors("Remi Cresson");

    // Input image
    AddParameter(ParameterType_InputImage, "in", "input image");

    // Input no-data value
    AddParameter(ParameterType_Float, "nodata", "nodata value");
    MandatoryOn                      ("nodata");
    SetDefaultParameterFloat         ("nodata", 0);

    // Grid
    AddParameter(ParameterType_Group, "grid", "grid settings");
    AddParameter(ParameterType_Int, "grid.step", "step between patches");
    AddParameter(ParameterType_Int, "grid.psize", "patches size");

    // Strategy
    AddParameter(ParameterType_Choice, "strategy", "Selection strategy for validation/training patches");
    AddChoice("strategy.chessboard", "fifty fifty, like a chess board");

    // Output points
    AddParameter(ParameterType_OutputVectorData, "outvec", "output set of points");


    AddRAMParameter();

  }


  void DoExecute()
  {

    // Compute no-data mask
    m_NoDataFilter = IsNoDataFilterType::New();
    m_NoDataFilter->GetFunctor().SetNoDataValue(GetParameterFloat("nodata"));
    m_NoDataFilter->UpdateOutputInformation();

    // Padding 1 pixel
    UInt8ImageType::SizeType size = m_NoDataFilter->GetOutput()->GetLargestPossibleRegion().GetSize();
    size[0] += 2;
    size[1] += 2;
    UInt8ImageType::SpacingType spacing = m_NoDataFilter->GetOutput()->GetSignedSpacing();
    UInt8ImageType::PointType origin = m_NoDataFilter->GetOutput()->GetOrigin();
    origin[0] -= spacing[0];
    origin[1] -= spacing[1];
    m_PadFilter = PadFilterType::New();
    m_PadFilter->SetInput( m_NoDataFilter->GetOutput() );
    m_PadFilter->SetOutputOrigin(origin);
    m_PadFilter->SetOutputSpacing(spacing);
    m_PadFilter->SetOutputSize(size);
    m_PadFilter->SetEdgePaddingValue( 0 );
    m_PadFilter->UpdateOutputInformation();

    // Morpho
    RadiusType rad;
    rad[0] = this->GetParameterInt("grid.psize")/2;
    rad[1] = this->GetParameterInt("grid.psize")/2;
    StructuringType se = StructuringType::Box(rad);
    m_MorphoFilter = MorphoFilterType::New();
    m_MorphoFilter->SetKernel(se);
    m_MorphoFilter->SetInput(m_PadFilter->GetOutput());
    m_MorphoFilter->SetForegroundValue(1);
    m_MorphoFilter->SetBackgroundValue(0);

    // Explicit streaming over the morphed mask, based on the RAM parameter
    typedef otb::RAMDrivenStrippedStreamingManager<UInt8ImageType> StreamingManagerType;
    StreamingManagerType::Pointer m_StreamingManager = StreamingManagerType::New();
    m_StreamingManager->SetAvailableRAMInMB(GetParameterInt("ram"));

    // We pad the image, if this is requested by the user
    UInt8ImageType::Pointer inputImage = m_MorphoFilter->GetOutput();
    UInt8ImageType::RegionType entireRegion = inputImage->GetLargestPossibleRegion();
    entireRegion.ShrinkByRadius(rad);
    m_StreamingManager->PrepareStreaming(inputImage, entireRegion );

    // First iteration to count the objects in each class
    int m_NumberOfDivisions = m_StreamingManager->GetNumberOfSplits();
    for (int m_CurrentDivision = 0; m_CurrentDivision < m_NumberOfDivisions; m_CurrentDivision++)
    {
      UInt8ImageType::RegionType streamRegion = m_StreamingManager->GetSplit(m_CurrentDivision);
      tf::PropagateRequestedRegion<UInt8ImageType>(inputImage, streamRegion);
      itk::ImageRegionConstIterator<UInt8ImageType> inIt (inputImage, streamRegion);
      for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt)
      {
        UInt8ImageType::InternalPixelType pixVal = inIt.Get();
        UInt8ImageType::IndexType idx = inIt.GetIndex();
        if (pixVal == 1)
        {
          // select this sample
        }
      }

    }
//
//    // Number of classes
//    const LabelImageType::InternalPixelType number_of_classes = class_end - class_begin + 1;
//
//    // Number of samples in each class (counted)
//    vnl_vector<IndexValueType> number_of_samples = tmp_number_of_samples.extract(number_of_classes, class_begin);
//
//    // Number of samples in each class (target)
//    vnl_vector<IndexValueType> target_number_of_samples(number_of_classes, 0);
//
//    otbAppLogINFO( "Number of classes: " << number_of_classes <<
//        " starting from " << class_begin <<
//        " to " << class_end << " (no-data is " << nodata << ")");
//    otbAppLogINFO( "Number of pixels in each class: " << number_of_samples );
//
//    // Check the smallest number of samples amongst classes
//    IndexValueType min_elem_in_class = itk::NumericTraits<IndexValueType>::max();
//    for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
//      min_elem_in_class = vcl_min(min_elem_in_class, number_of_samples[classIdx]);
//
//    // If one class is empty, throw an error
//    if (min_elem_in_class == 0)
//    {
//      otbAppLogFATAL("There is at least one class with no sample!")
//    }
//
//    // Sampling step for each classes
//    vnl_vector<IndexValueType> step_for_class(number_of_classes, 0);
//
//    // Compute the sampling step for each classes, depending on the chosen strategy
//    switch (this->GetParameterInt("strategy"))
//    {
//    // constant
//    case 0:
//    {
//      // Set the target number of samples in each class
//      target_number_of_samples.fill(GetParameterInt("strategy.constant.nb"));
//
//      // re adjust the number of samples to select in each class
//      if (min_elem_in_class < target_number_of_samples[0])
//      {
//        otbAppLogWARNING("Smallest class has " << min_elem_in_class <<
//            " samples but a number of " << target_number_of_samples[0] <<
//            " is given. Using " << min_elem_in_class);
//        target_number_of_samples.fill( min_elem_in_class );
//      }
//
//      // Compute the sampling step
//      for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
//        step_for_class[classIdx] = number_of_samples[classIdx] / target_number_of_samples[classIdx];
//    }
//    break;
//
//    // total
//    case 1:
//    {
//      // Compute the sampling step
//      IndexValueType step = number_of_samples.sum() / this->GetParameterInt("strategy.total.v");
//      if (step == 0)
//      {
//        otbAppLogWARNING("The number of samples available is smaller than the required number of samples. " <<
//            "Setting sampling step to 1.");
//        step = 1;
//      }
//      step_for_class.fill(step);
//
//      // Compute the target number of samples
//      for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
//        target_number_of_samples[classIdx] = number_of_samples[classIdx] / step;
//
//    }
//    break;
//
//    // smallest
//    case 2:
//    {
//      // Set the target number of samples to the smallest class
//      target_number_of_samples.fill( min_elem_in_class );
//
//      // Compute the sampling step
//      for (LabelImageType::InternalPixelType classIdx = 0 ; classIdx < number_of_classes ; classIdx++)
//        step_for_class[classIdx] = number_of_samples[classIdx] / target_number_of_samples[classIdx];
//
//    }
//    break;
//
//    // All
//    case 3:
//    {
//      // Easy
//      step_for_class.fill(1);
//      target_number_of_samples = number_of_samples;
//    }
//    break;
//    default:
//      otbAppLogFATAL("Strategy mode unknown :"<<this->GetParameterString("strategy"));
//      break;
//    }
//
//    // Print quick summary
//    otbAppLogINFO("Sampling summary:");
//    otbAppLogINFO("\tClass\tStep\tTot");
//    for (LabelImageType::InternalPixelType i = 0 ; i < number_of_classes ; i++)
//    {
//      vnl_vector<int> tmp (3,0);
//      tmp[0] = i + class_begin;
//      tmp[1] = step_for_class[i];
//      tmp[2] = target_number_of_samples[i];
//      otbAppLogINFO("\t" << tmp);
//    }
//
//    // Create a new vector data
//    // TODO: how to pre-allocate the datatree?
//    m_OutVectorData = VectorDataType::New();
//    DataTreeType::Pointer tree = m_OutVectorData->GetDataTree();
//    DataNodePointer root = tree->GetRoot()->Get();
//    DataNodePointer document = DataNodeType::New();
//    document->SetNodeType(DOCUMENT);
//    tree->Add(document, root);
//
//    // Duno if this makes sense?
//    m_OutVectorData->SetProjectionRef(inputImage->GetProjectionRef());
//    m_OutVectorData->SetOrigin(inputImage->GetOrigin());
//    m_OutVectorData->SetSpacing(inputImage->GetSpacing());
//
//    // Second iteration, to prepare the samples
//    vnl_vector<IndexValueType> sampledCount(number_of_classes, 0);
//    vnl_vector<IndexValueType> iteratorCount(number_of_classes, 0);
//    IndexValueType n_tot = 0;
//    const IndexValueType target_n_tot = target_number_of_samples.sum();
//    for (int m_CurrentDivision = 0; m_CurrentDivision < m_NumberOfDivisions; m_CurrentDivision++)
//    {
//      LabelImageType::RegionType streamRegion = m_StreamingManager->GetSplit(m_CurrentDivision);
//      tf::PropagateRequestedRegion<LabelImageType>(inputImage, streamRegion);
//      itk::ImageRegionConstIterator<LabelImageType> inIt (inputImage, streamRegion);
//
//      for (inIt.GoToBegin() ; !inIt.IsAtEnd() ; ++inIt)
//      {
//        LabelImageType::InternalPixelType classVal = inIt.Get();
//
//        if (classVal != nodata)
//        {
//          classVal -= class_begin;
//
//          // Update the current position
//          iteratorCount[classVal]++;
//
//          // Every Xi samples (Xi is the step for class i)
//          if (iteratorCount[classVal] % ((int) step_for_class[classVal]) == 0 &&
//              sampledCount[classVal] < target_number_of_samples[classVal])
//          {
//            // Add this sample
//            sampledCount[classVal]++;
//            n_tot++;
//            ShowProgress(n_tot, target_n_tot);
//
//            // Create a point
//            LabelImageType::PointType geo;
//            inputImage->TransformIndexToPhysicalPoint(inIt.GetIndex(), geo);
//            DataNodeType::PointType point;
//            point[0] = geo[0];
//            point[1] = geo[1];
//
//            // Add point to the VectorData tree
//            DataNodePointer newDataNode = DataNodeType::New();
//            newDataNode->SetPoint(point);
//            newDataNode->SetFieldAsInt("class", static_cast<int>(classVal));
//            tree->Add(newDataNode, document);
//
//          } // sample this one
//        }
//      } // next pixel
//    } // next streaming region
//    ShowProgressDone();
//
//    otbAppLogINFO( "Number of samples in each class: " << sampledCount );
//
//    otbAppLogINFO( "Writing output vector data");
//
//    SetParameterOutputVectorData("outvec", m_OutVectorData);

  }

private:
  IsNoDataFilterType::Pointer m_NoDataFilter;
  UInt8ImageType::Pointer m_PadFilter;
  MorphoFilterType::Pointer m_MorphoFilter;
  VectorDataType::Pointer m_OutVectorData;

}; // end of class

} // end namespace wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::PatchesSelection )
