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
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkMaskImageFilter.h"

// image utils
#include "otbTensorflowCommon.h"
#include "otbTensorflowSamplingUtils.h"
#include "itkImageRegionConstIteratorWithOnlyIndex.h"

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
  typedef itk::NearestNeighborInterpolateImageFunction<UInt8ImageType> NNInterpolatorType;

  typedef tf::Distribution<UInt8ImageType> DistributionType;

  typedef itk::MaskImageFilter<UInt8ImageType, UInt8ImageType, UInt8ImageType> MaskImageFilterType;

  void DoUpdateParameters()
  {
  }


  void DoInit()
  {

    // Documentation
    SetName("PatchesSelection");
    SetDescription("This application generate points sampled at regular interval over "
        "the input image region. The grid size and spacing can be configured.");
    SetDocLongDescription("This application produces a vector data containing "
        "a set of points centered on the patches lying in the valid regions of the input image. ");

    SetDocAuthors("Remi Cresson");

    // Input image
    AddParameter(ParameterType_InputImage, "in", "input image");
    AddParameter(ParameterType_InputImage, "mask", "input mask");
    MandatoryOff("mask");

    // Input no-data value
    AddParameter(ParameterType_Float, "nodata", "nodata value");
    MandatoryOn                      ("nodata");
    SetDefaultParameterFloat         ("nodata", 0);
    AddParameter(ParameterType_Bool,  "nocheck", "If on, no check on the validity of patches is performed");
    MandatoryOff                     ("nocheck");

    // Grid
    AddParameter(ParameterType_Group, "grid", "grid settings");
    AddParameter(ParameterType_Int, "grid.step", "step between patches");
    SetMinimumParameterIntValue    ("grid.step", 1);
    AddParameter(ParameterType_Int, "grid.psize", "patches size");
    SetMinimumParameterIntValue    ("grid.psize", 1);
    AddParameter(ParameterType_Int, "grid.offsetx", "offset of the grid (x axis)");
    SetDefaultParameterInt         ("grid.offsetx", 0);
    AddParameter(ParameterType_Int, "grid.offsety", "offset of the grid (y axis)");
    SetDefaultParameterInt         ("grid.offsety", 0);

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
      (void) point;
      black = true;
      (void) index;
    }
    ~SampleBundle(){}

    SampleBundle(const SampleBundle & other){
      dist = other.GetDistribution();
      id = other.GetSampleID();
      point = other.GetPosition();
      black = other.GetBlack();
      index = other.GetIndex();
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

    UInt8ImageType::IndexType& GetModifiableIndex()
    {
      return index;
    }

    UInt8ImageType::IndexType GetIndex() const
    {
      return index;
    }

  private:

    DistributionType dist;
    unsigned int id;
    DataNodePointType point;
    bool black;
    UInt8ImageType::IndexType index;
  };

  /*
   * Apply the given function at each sampling location, checking if the patch is valid or not
   */
  template<typename TLambda>
  void Apply(TLambda lambda)
  {

    int userOffX = GetParameterInt("grid.offsetx");
    int userOffY = GetParameterInt("grid.offsety");

    // Explicit streaming over the morphed mask, based on the RAM parameter
    typedef otb::RAMDrivenStrippedStreamingManager<UInt8ImageType> StreamingManagerType;
    StreamingManagerType::Pointer m_StreamingManager = StreamingManagerType::New();
    m_StreamingManager->SetAvailableRAMInMB(GetParameterInt("ram"));

    UInt8ImageType::Pointer inputImage;
    bool readInput = true;
    if (GetParameterInt("nocheck")==1)
      {
      otbAppLogINFO("\"nocheck\" mode is enabled. Input image pixels no-data values will not be checked.");
      if (HasValue("mask"))
        {
        otbAppLogINFO("Using the provided \"mask\" parameter.");
        inputImage = GetParameterUInt8Image("mask");
        }
      else
        {
        // This is just a hack to not trigger the whole morpho/pad pipeline
        inputImage = m_NoDataFilter->GetOutput();
        readInput = false;
        }
      }
    else
      {
      inputImage = m_MorphoFilter->GetOutput();

      // Offset update because the morpho filter pads the input image with 1 pixel border
      userOffX += 1;
      userOffY += 1;
      }
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

    // Offset update
    userOffX %= step ;
    userOffY %= step ;

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

        if (idx[0] % step == userOffX && idx[1] % step == userOffY)
        {
          UInt8ImageType::InternalPixelType pixVal = 1;
          if (readInput)
            pixVal = inIt.Get();

          if (pixVal == 1)
          {
            // Update grid position
            pos[0] = idx[0] / step;
            pos[1] = idx[1] / step;

            // Compute coordinates
            UInt8ImageType::PointType geo;
            inputImage->TransformIndexToPhysicalPoint(inIt.GetIndex(), geo);
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
  AllocateSamples(unsigned int nbOfClasses = 2)
  {
    // Nb of samples (maximum)
    const UInt8ImageType::RegionType entireRegion = m_NoDataFilter->GetOutput()->GetLargestPossibleRegion();
    const unsigned int maxNbOfCols = std::ceil(entireRegion.GetSize(0)/GetParameterInt("grid.step")) + 1;
    const unsigned int maxNbOfRows = std::ceil(entireRegion.GetSize(1)/GetParameterInt("grid.step")) + 1;
    unsigned int maxNbOfSamples = 1;
    maxNbOfSamples *= maxNbOfCols;
    maxNbOfSamples *= maxNbOfRows;

    // Nb of classes
    SampleBundle initSB(nbOfClasses);
    std::vector<SampleBundle> bundles(maxNbOfSamples, initSB);

    return bundles;
  }

  void SetBlackOrWhiteBundle(SampleBundle & bundle, unsigned int & count,
      const UInt8ImageType::IndexType & pos, const UInt8ImageType::PointType & geo)
  {
    // Black or white
    bool black = ((pos[0] + pos[1]) % 2 == 0);

    bundle.GetModifiableSampleID() = count;
    bundle.GetModifiablePosition() = geo;
    bundle.GetModifiableBlack() = black;
    bundle.GetModifiableIndex() = pos;
    count++;

  }

  /*
   * Samples are placed at regular intervals
   */
  void SampleChessboard()
  {

    std::vector<SampleBundle> bundles = AllocateSamples();

    unsigned int count = 0;
    auto lambda = [this, &count, &bundles]
                   (const UInt8ImageType::IndexType & pos, const UInt8ImageType::PointType & geo) {
      SetBlackOrWhiteBundle(bundles[count], count, pos, geo);
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

    std::vector<SampleBundle> bundles = AllocateSamples(GetParameterInt("strategy.balanced.nclasses"));

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
        SetBlackOrWhiteBundle(bundles[count], count, pos, geo);
      }
    };

    Apply(lambda);
    bundles.resize(count);

    otbAppLogINFO("Total number of candidates: " << count );

    // 2. Seed = spatially random samples

    otbAppLogINFO("Spatial sampling proportion " << GetParameterFloat("strategy.balanced.sp"));

    const int samplingStep = static_cast<int>(1.0 / std::sqrt(GetParameterFloat("strategy.balanced.sp")));

    otbAppLogINFO("Spatial sampling step " << samplingStep);

    float step = 0;
    std::vector<SampleBundle> seed(count);
    std::vector<SampleBundle> candidates(count);

    unsigned int seedCount = 0;
    unsigned int candidatesCount = 0;
    for (auto& d: bundles)
    {
      if (d.GetIndex()[0] % samplingStep + d.GetIndex()[1] % samplingStep == 0)
      {
        seed[seedCount] = d;
        seedCount++;
      }
      else
      {
        candidates[candidatesCount] = d;
        candidatesCount++;
      }
      step++;
    }

    seed.resize(seedCount);
    candidates.resize(candidatesCount);

    otbAppLogINFO("Spatial seed has " << seedCount << " samples");

    unsigned int nbToRemove = static_cast<unsigned int>(seedCount - GetParameterFloat("strategy.balanced.sp") * count);

    otbAppLogINFO("Adjust spatial seed removing " << nbToRemove << " samples");

    float removalRate = static_cast<float>(seedCount) / static_cast<float>(nbToRemove);
    float removalStep = 0;
    auto removeSamples = [&removalStep, &removalRate](SampleBundle & b) -> bool {
      (void) b;
      bool ret = false;
      if (removalStep >= removalRate)
        {
        removalStep = fmod(removalStep, removalRate);
        ret = true;
        }
      else
        ret = false;
      removalStep++;
      return ret;;
    };
    auto iterator = std::remove_if(seed.begin(), seed.end(), removeSamples);
    seed.erase(iterator, seed.end());

    otbAppLogINFO("Spatial seed size : " << seed.size());

    // 3. Compute seed distribution

    const unsigned int nbOfClasses = GetParameterInt("strategy.balanced.nclasses");
    DistributionType seedDist(nbOfClasses);
    for (auto& d: seed)
      seedDist.Update(d.GetDistribution());

    otbAppLogINFO("Spatial seed distribution: " << seedDist.ToString());

    // 4. Select other samples to feed the seed

    otbAppLogINFO("Balance seed candidates size: " << candidates.size());

    // Sort by cos
    auto comparator = [&seedDist](const SampleBundle & a, const SampleBundle & b) -> bool{
      return a.GetDistribution().Cosinus(seedDist) > b.GetDistribution().Cosinus(seedDist);
    };
    sort(candidates.begin(), candidates.end(), comparator);

    DistributionType idealDist(nbOfClasses, 1.0 / std::sqrt(static_cast<float>(nbOfClasses)));
    float minCos = 0;
    unsigned int samplesAdded = 0;
    seed.resize(seed.size()+candidates.size(), SampleBundle(nbOfClasses));
    while(candidates.size() > 0)
    {
      // Get the less correlated sample
      SampleBundle candidate = candidates.back();

      // Update distribution
      seedDist.Update(candidate.GetDistribution());

      // Compute cos of the updated distribution
      float idealCos = seedDist.Cosinus(idealDist);
      if (idealCos > minCos)
      {
        minCos = idealCos;
        seed[seedCount] = candidate;
        seedCount++;
        candidates.pop_back();
        samplesAdded++;
      }
      else
      {
        break;
      }
    }
    seed.resize(seedCount);

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
    otbAppLogINFO("Grid step : " << this->GetParameterInt("grid.step"));
    otbAppLogINFO("Patch size : " << this->GetParameterInt("grid.psize"));

    // Compute no-data mask
    m_NoDataFilter = IsNoDataFilterType::New();
    m_NoDataFilter->GetFunctor().SetNoDataValue(GetParameterFloat("nodata"));
    m_NoDataFilter->SetInput(GetParameterFloatVectorImage("in"));
    m_NoDataFilter->UpdateOutputInformation();
    UInt8ImageType::Pointer src = m_NoDataFilter->GetOutput();

    // If mask available, use it
    if (HasValue("mask"))
      {
      if (GetParameterUInt8Image("mask")->GetLargestPossibleRegion().GetSize() !=
          GetParameterFloatVectorImage("in")->GetLargestPossibleRegion().GetSize())
        otbAppLogFATAL("Mask must have the same size as the input image!");
      m_MaskImageFilter = MaskImageFilterType::New();
      m_MaskImageFilter->SetInput(m_NoDataFilter->GetOutput());
      m_MaskImageFilter->SetMaskImage(GetParameterUInt8Image("mask"));
      m_MaskImageFilter->UpdateOutputInformation();
      src = m_MaskImageFilter->GetOutput();
      }

    // Padding 1 pixel
    UInt8ImageType::SizeType size = src->GetLargestPossibleRegion().GetSize();
    size[0] += 2;
    size[1] += 2;
    UInt8ImageType::SpacingType spacing = src->GetSignedSpacing();
    UInt8ImageType::PointType origin = src->GetOrigin();
    origin[0] -= spacing[0];
    origin[1] -= spacing[1];
    m_PadFilter = PadFilterType::New();
    NNInterpolatorType::Pointer nnInterpolator = NNInterpolatorType::New();
    m_PadFilter->SetInterpolator(nnInterpolator);
    m_PadFilter->SetInput( src );
    m_PadFilter->SetOutputOrigin(origin);
    m_PadFilter->SetOutputSpacing(spacing);
    m_PadFilter->SetOutputSize(size);
    m_PadFilter->SetEdgePaddingValue( 0 );
    m_PadFilter->UpdateOutputInformation();

    // Morpho
    m_Radius[0] = this->GetParameterInt("grid.psize") / 2;
    m_Radius[1] = this->GetParameterInt("grid.psize") / 2;
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
  RadiusType                   m_Radius;
  IsNoDataFilterType::Pointer  m_NoDataFilter;
  PadFilterType::Pointer       m_PadFilter;
  MorphoFilterType::Pointer    m_MorphoFilter;
  VectorDataType::Pointer      m_OutVectorDataTrain;
  VectorDataType::Pointer      m_OutVectorDataValid;
  MaskImageFilterType::Pointer m_MaskImageFilter;
}; // end of class

} // end namespace wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT( otb::Wrapper::PatchesSelection )
