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
    SetMinimumParameterIntValue    ("grid.step", 1);
    AddParameter(ParameterType_Int, "grid.psize", "patches size");
    SetMinimumParameterIntValue    ("grid.psize", 1);

    // Strategy
    AddParameter(ParameterType_Choice, "strategy", "Selection strategy for validation/training patches");
    AddChoice("strategy.chessboard", "fifty fifty, like a chess board");

    // Output points
    AddParameter(ParameterType_OutputVectorData, "outtrain", "output set of points (training)");
    AddParameter(ParameterType_OutputVectorData, "outvalid", "output set of points (validation)");


    AddRAMParameter();

  }


  void DoExecute()
  {

    // Compute no-data mask
    m_NoDataFilter = IsNoDataFilterType::New();
    m_NoDataFilter->GetFunctor().SetNoDataValue(GetParameterFloat("nodata"));
    m_NoDataFilter->SetInput(GetParameterFloatVectorImage("in"));
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
    m_MorphoFilter->UpdateOutputInformation();

    // Prepare output vector data
    m_OutVectorDataTrain = VectorDataType::New();
    m_OutVectorDataValid = VectorDataType::New();
    m_OutVectorDataTrain->SetProjectionRef(m_MorphoFilter->GetOutput()->GetProjectionRef());
    m_OutVectorDataValid->SetProjectionRef(m_MorphoFilter->GetOutput()->GetProjectionRef());

    DataTreeType::Pointer treeTrain = m_OutVectorDataTrain->GetDataTree();
    DataTreeType::Pointer treeValid = m_OutVectorDataValid->GetDataTree();
    DataNodePointer rootTrain = treeTrain->GetRoot()->Get();
    DataNodePointer rootValid = treeValid->GetRoot()->Get();
    DataNodePointer documentTrain = DataNodeType::New();
    DataNodePointer documentValid = DataNodeType::New();
    documentTrain->SetNodeType(DOCUMENT);
    documentValid->SetNodeType(DOCUMENT);
    treeTrain->Add(documentTrain, rootTrain);
    treeValid->Add(documentValid, rootValid);

    // Explicit streaming over the morphed mask, based on the RAM parameter
    typedef otb::RAMDrivenStrippedStreamingManager<UInt8ImageType> StreamingManagerType;
    StreamingManagerType::Pointer m_StreamingManager = StreamingManagerType::New();
    m_StreamingManager->SetAvailableRAMInMB(GetParameterInt("ram"));

    UInt8ImageType::Pointer inputImage = m_MorphoFilter->GetOutput();
    UInt8ImageType::RegionType entireRegion = inputImage->GetLargestPossibleRegion();
    entireRegion.ShrinkByRadius(rad);
    m_StreamingManager->PrepareStreaming(inputImage, entireRegion );
    UInt8ImageType::IndexType start;
    start[0] = rad[0] + 1;
    start[1] = rad[1] + 1;

    int m_NumberOfDivisions = m_StreamingManager->GetNumberOfSplits();
    UInt8ImageType::IndexType pos;
    UInt8ImageType::IndexValueType step = GetParameterInt("grid.step");
    pos.Fill(0);
    bool black = true;
    unsigned int id = 0;
    for (int m_CurrentDivision = 0; m_CurrentDivision < m_NumberOfDivisions; m_CurrentDivision++)
    {
      otbAppLogINFO("Processing split " << (m_CurrentDivision+1) << "/" << m_NumberOfDivisions);

      UInt8ImageType::RegionType streamRegion = m_StreamingManager->GetSplit(m_CurrentDivision);
      tf::PropagateRequestedRegion<UInt8ImageType>(inputImage, streamRegion);
      itk::ImageRegionConstIterator<UInt8ImageType> inIt (inputImage, streamRegion);

      for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt)
      {
        UInt8ImageType::IndexType idx = inIt.GetIndex();
        idx[0] -= start[0];
        idx[1] -= start[1];

        if (idx[0] % step == 0 && idx[1] % step == 0)
        {
          UInt8ImageType::InternalPixelType pixVal = inIt.Get();

          if (pixVal == 1)
          {
            // Update grid position
            pos[0] = idx[0] / step;
            pos[1] = idx[1] / step;

            // First box
            black = ((pos[0] + pos[1]) % 2 == 0);

            // Compute coordinates
            UInt8ImageType::PointType geo;
            inputImage->TransformIndexToPhysicalPoint(inIt.GetIndex(), geo);
            DataNodeType::PointType point;
            point[0] = geo[0];
            point[1] = geo[1];

            // Add point to the VectorData tree
            DataNodePointer newDataNode = DataNodeType::New();
            newDataNode->SetPoint(point);
            newDataNode->SetFieldAsInt("id", id);
            id++;

            // select this sample
            if (black)
            {
              // Train
              treeTrain->Add(newDataNode, documentTrain);

            }
            else
            {
              // Valid
              treeValid->Add(newDataNode, documentValid);

            }
          }
        }
      }

    }

    otbAppLogINFO( "Writing output samples positions (Training)");

    SetParameterOutputVectorData("outtrain", m_OutVectorDataTrain);

    otbAppLogINFO( "Writing output samples positions (Validation)");

    SetParameterOutputVectorData("outvalid", m_OutVectorDataValid);

  }

private:
  IsNoDataFilterType::Pointer m_NoDataFilter;
  PadFilterType::Pointer      m_PadFilter;
  MorphoFilterType::Pointer   m_MorphoFilter;
  VectorDataType::Pointer     m_OutVectorDataTrain;
  VectorDataType::Pointer     m_OutVectorDataValid;

}; // end of class

} // end namespace wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::PatchesSelection )
