/*=========================================================================

  Copyright (c) 2018-2019 Remi Cresson (IRSTEA)
  Copyright (c) 2020-2021 Remi Cresson (INRAE)


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTensorflowGraphOperations.h"

namespace otb {
namespace tf {


//
// Load SavedModel variables
//
void RestoreModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle)
{
  tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<tensorflow::tstring>()() = path;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict =
  {{bundle.meta_graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
  auto status = bundle.session->Run(feed_dict, {}, {bundle.meta_graph_def.saver_def().restore_op_name()}, nullptr);
  if (!status.ok())
    {
    itkGenericExceptionMacro("Can't restore the input model: " << status.ToString() );
    }
}

//
// Save SavedModel variables
//
void SaveModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle)
{
  tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<tensorflow::tstring>()() = path;
  std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict =
  {{bundle.meta_graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
  auto status = bundle.session->Run(feed_dict, {}, {bundle.meta_graph_def.saver_def().save_tensor_name()}, nullptr);
  if (!status.ok())
    {
    itkGenericExceptionMacro("Can't restore the input model: " << status.ToString() );
    }
}

//
// Load a SavedModel
//
void LoadModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle, std::vector<std::string> tagList)
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
  auto status = tensorflow::LoadSavedModel(tensorflow::SessionOptions(), runoptions,
      path, tagSets, &bundle);
  if (!status.ok())
    {
    itkGenericExceptionMacro("Can't load the input model: " << status.ToString() );
    }

}

//
// Print a lot of stuff about the specified nodes of the graph
//
void PrintNodeAttributes(const tensorflow::GraphDef & graph, std::vector<std::string> & nodesNames)
{
  std::cout << "Go through graph:" << std::endl;
  std::cout << "#\tname" << std::endl;
  for (int i = 0 ; i < graph.node_size() ; i++)
  {
    tensorflow::NodeDef node = graph.node(i);
    std::cout << i << "\t" << node.name() << std::endl;

    for (std::vector<std::string>::iterator nameIt = nodesNames.begin();
        nameIt != nodesNames.end(); ++nameIt)
    {
      if (node.name().compare((*nameIt)) == 0)
      {
        std::cout << "Node " << i << " : " << std::endl;
        std::cout << "\tName: " << node.name() << std::endl;
        std::cout << "\tinput_size() : " << node.input_size() << std::endl;
        std::cout << "\tPrintDebugString --------------------------------";
        std::cout << std::endl;
        node.PrintDebugString();
        std::cout << "\t-------------------------------------------------" << std::endl;

        // display all attributes of the node
        std::cout << "\tAttributes of the node: " << std::endl;
        for (auto attr = node.attr().begin() ; attr != node.attr().end() ; attr++)
        {
          std::cout << "\t\tKey :" << attr->first << std::endl;
          std::cout << "\t\tValue.value_case() :" << attr->second.value_case() << std::endl;
          std::cout << "\t\tPrintDebugString --------------------------------";
          std::cout << std::endl;
          attr->second.PrintDebugString();
          std::cout << "\t\t-------------------------------------------------" << std::endl;
          std::cout << std::endl;
        } // next attribute
      } // node name match
    } // next node name
  } // next node of the graph

}

} // end namespace tf
} // end namespace otb
