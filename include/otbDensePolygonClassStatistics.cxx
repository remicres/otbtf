/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


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

// Filters
#include "otbStatisticsXMLFileWriter.h"
#include "otbWrapperElevationParametersHandler.h"
#include "otbVectorDataToLabelImageFilter.h"
#include "otbImageToNoDataMaskFilter.h"
#include "otbStreamingStatisticsMapFromLabelImageFilter.h"
#include "otbVectorDataIntoImageProjectionFilter.h"
#include "otbImageToVectorImageCastFilter.h"

// OGR
#include "otbOGR.h"

namespace otb
{
namespace Wrapper
{
/** Utility function to negate std::isalnum */
bool IsNotAlphaNum(char c)
  {
  return !std::isalnum(c);
  }

class DensePolygonClassStatistics : public Application
{
public:
  /** Standard class typedefs. */
  typedef DensePolygonClassStatistics   Self;
  typedef Application                   Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Standard macro */
  itkNewMacro(Self);
  itkTypeMacro(DensePolygonClassStatistics, Application);

  /** DataObjects typedef */
  typedef UInt32ImageType                           LabelImageType;
  typedef UInt8ImageType                            MaskImageType;
  typedef VectorData<>                              VectorDataType;

  /** ProcessObjects typedef */
  typedef otb::VectorDataIntoImageProjectionFilter<VectorDataType,
      FloatVectorImageType>                                                       VectorDataReprojFilterType;

  typedef otb::VectorDataToLabelImageFilter<VectorDataType, LabelImageType>       RasterizeFilterType;

  typedef otb::VectorImage<MaskImageType::PixelType>                              InternalMaskImageType;
  typedef otb::ImageToNoDataMaskFilter<FloatVectorImageType, MaskImageType>       NoDataMaskFilterType;
  typedef otb::ImageToVectorImageCastFilter<MaskImageType, InternalMaskImageType> CastFilterType;

  typedef otb::StreamingStatisticsMapFromLabelImageFilter<InternalMaskImageType,
      LabelImageType>                                                             StatsFilterType;

  typedef otb::StatisticsXMLFileWriter<FloatVectorImageType::PixelType>           StatWriterType;

  void DoInit()
  {
    SetName("DensePolygonClassStatistics");
    SetDescription("Computes statistics on a training polygon set.");

    // Documentation
    SetDocLongDescription("The application processes a dense set of polygons "
      "intended for training (they should have a field giving the associated "
      "class). The geometries are analyzed against a support image to compute "
      "statistics : \n"
      "  - number of samples per class\n"
      "  - number of samples per geometry\n");
    SetDocLimitations("None");
    SetDocAuthors("Remi Cresson");

    AddDocTag(Tags::Learning);

    AddParameter(ParameterType_InputImage,  "in",   "Input image");
    SetParameterDescription("in", "Support image that will be classified");
    
    AddParameter(ParameterType_InputVectorData, "vec", "Input vectors");
    SetParameterDescription("vec","Input geometries to analyze");
    
    AddParameter(ParameterType_OutputFilename, "out", "Output XML statistics file");
    SetParameterDescription("out","Output file to store statistics (XML format)");

    AddParameter(ParameterType_ListView, "field", "Field Name");
    SetParameterDescription("field","Name of the field carrying the class number in the input vectors.");
    SetListViewSingleSelectionMode("field",true);

    ElevationParametersHandler::AddElevationParameters(this, "elev");

    AddRAMParameter();

    // Doc example parameter settings
    SetDocExampleParameterValue("in", "support_image.tif");
    SetDocExampleParameterValue("vec", "variousVectors.shp");
    SetDocExampleParameterValue("field", "label");
    SetDocExampleParameterValue("out","polygonStat.xml");

  }

  void DoExecute()
  {

  // Retrieve the field name
  std::vector<int> selectedCFieldIdx = GetSelectedItems("field");

  if(selectedCFieldIdx.empty())
    {
    otbAppLogFATAL(<<"No field has been selected for data labelling!");
    }

  std::vector<std::string> cFieldNames = GetChoiceNames("field");  
  std::string fieldName = cFieldNames[selectedCFieldIdx.front()];

  otb::Wrapper::ElevationParametersHandler::SetupDEMHandlerFromElevationParameters(this,"elev");

  // Get inputs
  FloatVectorImageType::Pointer xs = GetParameterImage("in");
  VectorDataType* shp = GetParameterVectorData("vec");

  // Reproject vector data
  m_VectorDataReprojectionFilter = VectorDataReprojFilterType::New();
  m_VectorDataReprojectionFilter->SetInputVectorData(shp);
  m_VectorDataReprojectionFilter->SetInputImage(xs);
  m_VectorDataReprojectionFilter->Update();

  // Internal no-data value
  const LabelImageType::ValueType intNoData =
      itk::NumericTraits<LabelImageType::ValueType>::max();

  // Rasterize vector data (geometry ID)
  m_RasterizeFIDFilter = RasterizeFilterType::New();
  m_RasterizeFIDFilter->AddVectorData(m_VectorDataReprojectionFilter->GetOutput());
  m_RasterizeFIDFilter->SetOutputOrigin(xs->GetOrigin());
  m_RasterizeFIDFilter->SetOutputSpacing(xs->GetSignedSpacing());
  m_RasterizeFIDFilter->SetOutputSize(xs->GetLargestPossibleRegion().GetSize());
  m_RasterizeFIDFilter->SetBurnAttribute("________"); // Trick to get the polygon ID
  m_RasterizeFIDFilter->SetGlobalWarningDisplay(false);
  m_RasterizeFIDFilter->SetOutputProjectionRef(xs->GetProjectionRef());
  m_RasterizeFIDFilter->SetBackgroundValue(intNoData);
  m_RasterizeFIDFilter->SetDefaultBurnValue(0);

  // Rasterize vector data (geometry class)
  m_RasterizeClassFilter = RasterizeFilterType::New();
  m_RasterizeClassFilter->AddVectorData(m_VectorDataReprojectionFilter->GetOutput());
  m_RasterizeClassFilter->SetOutputOrigin(xs->GetOrigin());
  m_RasterizeClassFilter->SetOutputSpacing(xs->GetSignedSpacing());
  m_RasterizeClassFilter->SetOutputSize(xs->GetLargestPossibleRegion().GetSize());
  m_RasterizeClassFilter->SetBurnAttribute(fieldName);
  m_RasterizeClassFilter->SetOutputProjectionRef(xs->GetProjectionRef());
  m_RasterizeClassFilter->SetBackgroundValue(intNoData);
  m_RasterizeClassFilter->SetDefaultBurnValue(0);

  // No data mask
  m_NoDataFilter = NoDataMaskFilterType::New();
  m_NoDataFilter->SetInput(xs);
  m_NoDataCastFilter = CastFilterType::New();
  m_NoDataCastFilter->SetInput(m_NoDataFilter->GetOutput());

  // Stats (geometry ID)
  m_FIDStatsFilter = StatsFilterType::New();
  m_FIDStatsFilter->SetInput(m_NoDataCastFilter->GetOutput());
  m_FIDStatsFilter->SetInputLabelImage(m_RasterizeFIDFilter->GetOutput());
  m_FIDStatsFilter->GetStreamer()->SetAutomaticAdaptativeStreaming(GetParameterInt("ram"));
  AddProcess(m_FIDStatsFilter->GetStreamer(), "Computing number of samples per vector");
  m_FIDStatsFilter->Update();

  // Stats (geometry class)
  m_ClassStatsFilter = StatsFilterType::New();
  m_ClassStatsFilter->SetInput(m_NoDataCastFilter->GetOutput());
  m_ClassStatsFilter->SetInputLabelImage(m_RasterizeClassFilter->GetOutput());
  m_ClassStatsFilter->GetStreamer()->SetAutomaticAdaptativeStreaming(GetParameterInt("ram"));
  AddProcess(m_ClassStatsFilter->GetStreamer(), "Computing number of samples per class");
  m_ClassStatsFilter->Update();

  // Remove the no-data entries
  StatsFilterType::LabelPopulationMapType fidMap = m_FIDStatsFilter->GetLabelPopulationMap();
  StatsFilterType::LabelPopulationMapType classMap = m_ClassStatsFilter->GetLabelPopulationMap();
  fidMap.erase(intNoData);
  classMap.erase(intNoData);

  m_StatWriter = StatWriterType::New();
  m_StatWriter->SetFileName(this->GetParameterString("out"));
  m_StatWriter->AddInputMap<StatsFilterType::LabelPopulationMapType>("samplesPerClass", classMap);
  m_StatWriter->AddInputMap<StatsFilterType::LabelPopulationMapType>("samplesPerVector", fidMap);
  m_StatWriter->Update();

  }

  void DoUpdateParameters()
  {
     if (HasValue("vec"))
      {
      std::string vectorFile = GetParameterString("vec");
      ogr::DataSource::Pointer ogrDS =
        ogr::DataSource::New(vectorFile, ogr::DataSource::Modes::Read);
      ogr::Layer layer = ogrDS->GetLayer(0);
      ogr::Feature feature = layer.ogr().GetNextFeature();

      ClearChoices("field");

      for(int iField=0; iField<feature.ogr().GetFieldCount(); iField++)
        {
        std::string key, item = feature.ogr().GetFieldDefnRef(iField)->GetNameRef();
        key = item;
        std::string::iterator end = std::remove_if(key.begin(),key.end(),IsNotAlphaNum);
        std::transform(key.begin(), end, key.begin(), tolower);

        OGRFieldType fieldType = feature.ogr().GetFieldDefnRef(iField)->GetType();

        if(fieldType == OFTString || fieldType == OFTInteger || fieldType == OFTInteger64)
          {
          std::string tmpKey="field."+key.substr(0, end - key.begin());
          AddChoice(tmpKey,item);
          }
        }
      }

     // Check that the extension of the output parameter is XML (mandatory for
     // StatisticsXMLFileWriter)
     // Check it here to trigger the error before polygons analysis

     if (HasValue("out"))
       {
       // Store filename extension
       // Check that the right extension is given : expected .xml
       const std::string extension = itksys::SystemTools::GetFilenameLastExtension(this->GetParameterString("out"));

       if (itksys::SystemTools::LowerCase(extension) != ".xml")
         {
         otbAppLogFATAL( << extension << " is a wrong extension for parameter \"out\": Expected .xml" );
         }
       }
  }



private:
  // Filters
  VectorDataReprojFilterType::Pointer m_VectorDataReprojectionFilter;
  RasterizeFilterType::Pointer m_RasterizeFIDFilter;
  RasterizeFilterType::Pointer m_RasterizeClassFilter;
  NoDataMaskFilterType::Pointer m_NoDataFilter;
  CastFilterType::Pointer m_NoDataCastFilter;
  StatsFilterType::Pointer m_FIDStatsFilter;
  StatsFilterType::Pointer m_ClassStatsFilter;
  StatWriterType::Pointer m_StatWriter;

};

} // end of namespace Wrapper
} // end of namespace otb

OTB_APPLICATION_EXPORT(otb::Wrapper::DensePolygonClassStatistics)
