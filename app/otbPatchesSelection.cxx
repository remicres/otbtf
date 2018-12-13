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
#include "otbTensorflowSamplingUtils.h"

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
  typedef DataNodeType::PointType                      DataNodePointType;

  /** typedefs */
  typedef IsNoData<FloatVectorImageType::PixelType, UInt8ImageType::PixelType > IsNoDataFunctorType;
  typedef itk::UnaryFunctorImageFilter<FloatVectorImageType, UInt8ImageType, IsNoDataFunctorType> IsNoDataFilterType;

  typedef itk::FlatStructuringElement<2>                                         StructuringType;
  typedef StructuringType::RadiusType                                            RadiusType;

  typedef itk::BinaryErodeImageFilter<UInt8ImageType, UInt8ImageType, StructuringType> MorphoFilterType;

  typedef otb::StreamingResampleImageFilter<UInt8ImageType,UInt8ImageType> PadFilterType;

  typedef tf::Distribution<UInt8ImageType> DistributionType;

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
    AddChoice("strategy.balanced", "you can chose the degree of spatial randomness vs class balance");
    AddParameter(ParameterType_Float, "strategy.balanced.sp", "Spatial proportion: between 0 and 1, "
        "indicating the amount of randomly sampled data in space");
    SetMinimumParameterFloatValue    ("strategy.balanced.sp", 0);
    SetMaximumParameterFloatValue    ("strategy.balanced.sp", 1);
    SetDefaultParameterFloat         ("strategy.balanced.sp", 0.25);
    AddParameter(ParameterType_Int,   "strategy.balanced.nclasses", "Number of classes");
    SetMinimumParameterIntValue      ("strategy.balanced.nclasses", 2);
    MandatoryOn                      ("strategy.balanced.nclasses");
    AddParameter(ParameterType_InputImage, "strategy.balanced.labelimage", "input label image");
    MandatoryOn                           ("strategy.balanced.labelimage");

    // Output points
    AddParameter(ParameterType_OutputVectorData, "outtrain", "output set of points (training)");
    AddParameter(ParameterType_OutputVectorData, "outvalid", "output set of points (validation)");

    AddRAMParameter();

  }

  class SampleBundle
  {
  public:
    SampleBundle(){}
    SampleBundle(unsigned int nClasses){
      dist = DistributionType(nClasses);
      id = 0;
      black = true;
    }
    ~SampleBundle(){}

    SampleBundle(const SampleBundle & other){
      dist = other.GetDistribution();
      id = other.GetSampleID();
      point = other.GetPosition();
      black = other.GetBlack();
    }

    DistributionType GetDistribution() const
    {
      return dist;
    }

    DistributionType& GetModifiableDistribution()
    {
      return dist;
    }

    unsigned int GetSampleID() const
    {
      return id;
    }

    unsigned int& GetModifiableSampleID()
    {
      return id;
    }

    DataNodePointType GetPosition() const
    {
      return point;
    }

    DataNodePointType& GetModifiablePosition()
    {
      return point;
    }

    bool& GetModifiableBlack()
    {
      return black;
    }

    bool GetBlack() const
    {
      return black;
    }

  private:

    DistributionType dist;
    unsigned int id;
    DataNodePointType point;
    bool black;
  };

  /*
   * Apply the given function at each sampling location
   */
  template<typename TLambda>
  void Apply(TLambda lambda)
  {

    // Explicit streaming over the morphed mask, based on the RAM parameter
    typedef otb::RAMDrivenStrippedStreamingManager<UInt8ImageType> StreamingManagerType;
    StreamingManagerType::Pointer m_StreamingManager = StreamingManagerType::New();
    m_StreamingManager->SetAvailableRAMInMB(GetParameterInt("ram"));

    UInt8ImageType::Pointer inputImage = m_MorphoFilter->GetOutput();
    UInt8ImageType::RegionType entireRegion = inputImage->GetLargestPossibleRegion();
    entireRegion.ShrinkByRadius(m_Radius);
    m_StreamingManager->PrepareStreaming(inputImage, entireRegion );
    UInt8ImageType::IndexType start;
    start[0] = m_Radius[0] + 1;
    start[1] = m_Radius[1] + 1;

    int m_NumberOfDivisions = m_StreamingManager->GetNumberOfSplits();
    UInt8ImageType::IndexType pos;
    UInt8ImageType::IndexValueType step = GetParameterInt("grid.step");
    pos.Fill(0);
    for (int m_CurrentDivision = 0; m_CurrentDivision < m_NumberOfDivisions; m_CurrentDivision++)
    {
      otbAppLogINFO("Processing split " << (m_CurrentDivision + 1) << "/" << m_NumberOfDivisions);

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

            // Compute coordinates
            UInt8ImageType::PointType geo;
            m_MorphoFilter->GetOutput()->TransformIndexToPhysicalPoint(inIt.GetIndex(), geo);
            DataNodeType::PointType point;
            point[0] = geo[0];
            point[1] = geo[1];

            // Lambda call
            lambda(pos, geo);
          }
        }
      }

    }
  }

  /*
   * Allocate a std::vector of sample bundle
   */
  std::vector<SampleBundle>
  AllocateSamples()
  {
    // Nb of samples (maximum)
    const UInt8ImageType::RegionType entireRegion = m_MorphoFilter->GetOutput()->GetLargestPossibleRegion();
    const unsigned int maxNbOfCols = std::ceil(entireRegion.GetSize(0)/GetParameterInt("grid.step")) + 1;
    const unsigned int maxNbOfRows = std::ceil(entireRegion.GetSize(1)/GetParameterInt("grid.step")) + 1;
    unsigned int maxNbOfSamples = 1;
    maxNbOfSamples *= maxNbOfCols;
    maxNbOfSamples *= maxNbOfRows;

    // Nb of classes
    const unsigned int nbOfClasses = GetParameterInt("strategy.balanced.nclasses");
    SampleBundle initSB(nbOfClasses);
    std::vector<SampleBundle> bundles(maxNbOfSamples, initSB);

    return bundles;
  }

  /*
   * Samples are placed at regular intervals
   */
  void SampleChessboard()
  {

    std::vector<SampleBundle> bundles = AllocateSamples();

    unsigned int count = 0;
    auto lambda = [&count, &bundles]
                   (const UInt8ImageType::IndexType & pos, const UInt8ImageType::PointType & geo) {

      // Black or white
      bool black = ((pos[0] + pos[1]) % 2 == 0);

      bundles[count].GetModifiableSampleID() = count;
      bundles[count].GetModifiablePosition() = geo;
      bundles[count].GetModifiableBlack() = black;
      count++;
    };

    Apply(lambda);
    bundles.resize(count);

    // Export training/validation samples
    PopulateVectorData(bundles);
  }

  void SampleBalanced()
  {

    // 1. Compute distribution of all samples

    otbAppLogINFO("Computing samples distribution...");

    std::vector<SampleBundle> bundles = AllocateSamples();

    // Patch size
    UInt8ImageType::SizeType patchSize;
    patchSize.Fill(GetParameterInt("grid.psize"));
    unsigned int count = 0;
    auto lambda = [this, &bundles, &patchSize, &count]
                   (const UInt8ImageType::IndexType & pos, const UInt8ImageType::PointType & geo) {

      // Update this sample distribution
      if (tf::UpdateDistributionFromPatch<UInt8ImageType>(GetParameterUInt8Image("strategy.balanced.labelimage"),
          geo, patchSize, bundles[count].GetModifiableDistribution()))
      {
        bundles[count].GetModifiableSampleID() = count;
        bundles[count].GetModifiablePosition() = geo;
        bundles[count].GetModifiableBlack() = ((pos[0] + pos[1]) % 2 == 0);
        count++;
      }
    };

    Apply(lambda);
    bundles.resize(count);

    otbAppLogINFO("Total number of candidates: " << count );

    // 2. Seed = spatially random samples

    const float samplingStep = 1.0 / std::sqrt(GetParameterFloat("strategy.balanced.sp"));
    float step = 0;
    std::vector<SampleBundle> seed;
    std::vector<SampleBundle> candidates;

    for (auto& d: bundles)
    {
      if (step >= samplingStep)
      {
        seed.push_back(d);
        step = fmod(step, samplingStep);
      }
      else
      {
        candidates.push_back(d);
        step++;
      }
    }

    otbAppLogINFO("Spatial seed size : " << seed.size());

    // 3. Compute seed distribution

    const unsigned int nbOfClasses = GetParameterInt("strategy.balanced.nclasses");
    DistributionType seedDist(nbOfClasses);
    for (auto& d: seed)
      seedDist.Update(d.GetDistribution());

    otbAppLogINFO("Spatial seed distribution: " << seedDist.ToString());

    // 4. Select other samples to feed the seed

    otbAppLogINFO("Balance seed candidates size: " << candidates.size());

    // Lambda for sorting
    auto comparator = [&seedDist](const SampleBundle & a, const SampleBundle & b) -> bool{
      return a.GetDistribution().Cosinus(seedDist) > b.GetDistribution().Cosinus(seedDist);
    };

    DistributionType idealDist(nbOfClasses, 1.0 / std::sqrt(static_cast<float>(nbOfClasses)));
    float minCos = 0;
    unsigned int samplesAdded = 0;
    while(candidates.size() > 0)
    {
      // Sort by cos
      sort(candidates.begin(), candidates.end(), comparator);

      // Get the less correlated sample
      SampleBundle candidate = candidates.back();

      // Update distribution
      seedDist.Update(candidate.GetDistribution());

      // Compute cos of the updated distribution
      float idealCos = seedDist.Cosinus(idealDist);
      if (idealCos > minCos)
      {
        minCos = idealCos;
        seed.push_back(candidate);
        candidates.pop_back();
        samplesAdded++;
      }
      else
      {
        break;
      }
    }

    otbAppLogINFO("Final samples number: " << seed.size() << " (" << samplesAdded << " samples added)");
    otbAppLogINFO("Final samples distribution: " << seedDist.ToString());

    // 5. Export training/validation samples
    PopulateVectorData(seed);
  }

  void PopulateVectorData(std::vector<SampleBundle> & samples)
  {
    // Get data tree
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

    unsigned int id = 0;
    for (const auto& sample: samples)
    {
      // Add point to the VectorData tree
      DataNodePointer newDataNode = DataNodeType::New();
      newDataNode->SetPoint(sample.GetPosition());
      newDataNode->SetFieldAsInt("id", id);
      id++;

      // select this sample
      if (sample.GetBlack())
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
    m_Radius[0] = this->GetParameterInt("grid.psize")/2;
    m_Radius[1] = this->GetParameterInt("grid.psize")/2;
    StructuringType se = StructuringType::Box(m_Radius);
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

    if (GetParameterAsString("strategy") == "chessboard")
    {
      otbAppLogINFO("Sampling at regular interval in space (\"Chessboard\" like)");

      SampleChessboard();
    }
    else if (GetParameterAsString("strategy") == "balanced")
    {
      otbAppLogINFO("Sampling with balancing strategy");

      SampleBalanced();
    }

    otbAppLogINFO( "Writing output samples positions");

    SetParameterOutputVectorData("outtrain", m_OutVectorDataTrain);
    SetParameterOutputVectorData("outvalid", m_OutVectorDataValid);

  }

private:
  RadiusType                  m_Radius;
  IsNoDataFilterType::Pointer m_NoDataFilter;
  PadFilterType::Pointer      m_PadFilter;
  MorphoFilterType::Pointer   m_MorphoFilter;
  VectorDataType::Pointer     m_OutVectorDataTrain;
  VectorDataType::Pointer     m_OutVectorDataValid;

}; // end of class

} // end namespace wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::PatchesSelection )
