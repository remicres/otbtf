/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTensorflowGraphOperations.h"

namespace otb
{
namespace tf
{


//
// Load SavedModel variables
//
void
RestoreModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle)
{
  tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<tensorflow::tstring>()() = path;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
    { bundle.meta_graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }
  };
  auto status = bundle.session->Run(feed_dict, {}, { bundle.meta_graph_def.saver_def().restore_op_name() }, nullptr);
  if (!status.ok())
  {
    itkGenericExceptionMacro("Can't restore the input model: " << status.ToString());
  }
}

//
// Save SavedModel variables
//
void
SaveModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle)
{
  tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<tensorflow::tstring>()() = path;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {
    { bundle.meta_graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }
  };
  auto status = bundle.session->Run(feed_dict, {}, { bundle.meta_graph_def.saver_def().save_tensor_name() }, nullptr);
  if (!status.ok())
  {
    itkGenericExceptionMacro("Can't restore the input model: " << status.ToString());
  }
}

//
// Load a SavedModel
//
void
LoadModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle, std::vector<std::string> tagList)
{
  // If the tag list is empty, we push back the default tag for model serving
  if (tagList.size() == 0)
    tagList.push_back(tensorflow::kSavedModelTagServe);

  // std::vector --> std::unordered_list
  std::unordered_set<std::string> tagSets;
  std::copy(tagList.begin(), tagList.end(), std::inserter(tagSets, tagSets.end())); // copy in unordered_set

  // Call to tensorflow::LoadSavedModel
  tensorflow::RunOptions runoptions;
  runoptions.set_trace_level(tensorflow::RunOptions_TraceLevel_FULL_TRACE);
  auto status = tensorflow::LoadSavedModel(tensorflow::SessionOptions(), runoptions, path, tagSets, &bundle);
  if (!status.ok())
  {
    itkGenericExceptionMacro("Can't load the input model: " << status.ToString());
  }
}


// Get the following attributes of the specified tensors (by name) of a graph:
// - layer name, as specified in the model
// - shape
// - datatype
void
GetTensorAttributes(const tensorflow::protobuf::Map<std::string, tensorflow::TensorInfo> layers,
                    std::vector<std::string> &                                           tensorsNames,
                    std::vector<std::string> &                                           layerNames,
                    std::vector<tensorflow::TensorShapeProto> &                          shapes,
                    std::vector<tensorflow::DataType> &                                  dataTypes,
                    std::vector<std::string>                                             blackList = {})
{
  // Clear shapes, datatypes, and layers names
  shapes.clear();
  dataTypes.clear();
  layerNames.clear();

  // Debug infos
  otbLogMacro(Debug, << "Nodes contained in the model:");
  for (auto const & layer : layers)
    otbLogMacro(Debug, << "\t" << layer.first);

  // Sort nodes names alphabetically
  std::size_t                              k = 0;             // Keep track of the indices in the protobuf map
  std::vector<std::pair<std::string, int>> sortedLayersNames; // vector of (name, index) pairs
  for (auto const & layer : layers)
  {
    // We exclude names, if any in the blacklist. Useful to avoid confusion
    // between user placeholders (aka constants) and input tensors.
    // Not used for output tensors.
    if (std::count(blackList.begin(), blackList.end(), layer.first) == 0)
    {
      sortedLayersNames.emplace_back(layer.first, k);
    }
    k++;
  }
  std::sort(sortedLayersNames.begin(), sortedLayersNames.end());
  otbLogMacro(Debug, << "Sorted (alphabetically) nodes names:");
  for (auto const & name : sortedLayersNames)
    otbLogMacro(Debug, << "\t" << name.first << " (index: " << name.second << ")");

  // When the user doesn't specify output.names, tensorsNames defaults to an empty list that we can not iterate over.
  // We change it to a list containing an empty string [""]
  if (tensorsNames.size() == 0)
  {
    otbLogMacro(Debug, << "No output.name specified. Using a default list with one empty string.");
    tensorsNames.push_back("");
  }

  // Next, we fill layerNames
  k = 0; // counter used for tensorsNames
  for (auto const & name : tensorsNames)
  {
    bool                   found = false;
    tensorflow::TensorInfo tensor_info;

    // If the user didn't specify the placeholdername, choose the kth layer inside the model
    if (name.size() == 0)
    {
      found = true;
      // select the k-th element of `layers` names, alphabeticallly sorted
      const std::string kthName = sortedLayersNames[k].first;
      auto              it = layers.begin();
      const int         kthIndex = sortedLayersNames[k].second;
      std::advance(it, kthIndex);
      layerNames.push_back(it->second.name());
      tensor_info = it->second;
      if (sortedLayersNames.size() > 1)
        otbLogMacro(Warning,
                    << "The provided tensor name is empty, and there are multiple available candidates in the graph. "
                       "Available tensors names from the graph have been sorted alphabetically, and the tensor #"
                    << kthIndex << " (aka \"" << it->first << "\") will be used. ");
    }

    // Else, if the user specified the placeholdername, find the corresponding layer inside the model
    else
    {
      otbLogMacro(Debug, << "Searching for corresponding node of \"" << name << "\"... ");
      for (auto const & layer : layers)
      {
        // layer is a pair (name, tensor_info)
        // cf https://stackoverflow.com/questions/63181951/how-to-get-graph-or-graphdef-from-a-given-model
        std::string layername = layer.first;
        if (layername.substr(0, layername.find(":")).compare(name) == 0)
        {
          found = true;
          layerNames.push_back(layer.second.name());
          tensor_info = layer.second;
          otbLogMacro(Debug, << "Found: " << layer.second.name() << " in the model");
        }
      } // next layer
    }   // end else

    k += 1;

    if (!found)
    {
      itkGenericExceptionMacro("Tensor name \"" << name << "\" not found. \n"
                                                << "You can list all inputs/outputs of your SavedModel by "
                                                << "running: \n\t `saved_model_cli show --dir your_model_dir --all`");
    }

    // Default tensor type
    tensorflow::DataType ts_dt = tensor_info.dtype();
    dataTypes.push_back(ts_dt);

    // Get the tensor's shape
    // Here we assure it's a tensor, with 1 shape
    tensorflow::TensorShapeProto ts_shp = tensor_info.tensor_shape();
    shapes.push_back(ts_shp);
  } // next tensor name
}

//
// Print a lot of stuff about the specified nodes of the graph
//
void
PrintNodeAttributes(const tensorflow::GraphDef & graph, const std::vector<std::string> & nodesNames)
{
  std::cout << "Go through graph:" << std::endl;
  std::cout << "#\tname" << std::endl;
  for (int i = 0; i < graph.node_size(); i++)
  {
    tensorflow::NodeDef node = graph.node(i);
    std::cout << i << "\t" << node.name() << std::endl;

    for (auto const & name : nodesNames)
    {
      if (node.name().compare(name) == 0)
      {
        std::cout << "Node " << i << " : " << std::endl;
        std::cout << "\tName: " << node.name() << std::endl;
        std::cout << "\tinput_size(): " << node.input_size() << std::endl;
        std::cout << "\tPrintDebugString --------------------------------";
        std::cout << std::endl;
        node.PrintDebugString();
        std::cout << "\t-------------------------------------------------" << std::endl;

        // display all attributes of the node
        std::cout << "\tAttributes of the node: " << std::endl;
        for (auto attr = node.attr().begin(); attr != node.attr().end(); attr++)
        {
          std::cout << "\t\tKey: " << attr->first << std::endl;
          std::cout << "\t\tValue.value_case(): " << attr->second.value_case() << std::endl;
          std::cout << "\t\tPrintDebugString --------------------------------";
          std::cout << std::endl;
          attr->second.PrintDebugString();
          std::cout << "\t\t-------------------------------------------------" << std::endl;
          std::cout << std::endl;
        } // next attribute
      }   // node name match
    }     // next node name
  }       // next node of the graph
}

} // end namespace tf
} // end namespace otb
