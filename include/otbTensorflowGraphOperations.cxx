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
// Restore a model from a path
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
// Restore a model from a path
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
// Load a session and a graph from a folder
//
void LoadModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle)
{

  tensorflow::RunOptions runoptions;
  runoptions.set_trace_level(tensorflow::RunOptions_TraceLevel_FULL_TRACE);
  auto status = tensorflow::LoadSavedModel(tensorflow::SessionOptions(), runoptions,
      path, {tensorflow::kSavedModelTagServe}, &bundle);
  if (!status.ok())
    {
    itkGenericExceptionMacro("Can't load the input model: " << status.ToString() );
    }

}

//
// Load a graph from a .meta file
//
tensorflow::GraphDef LoadGraph(std::string filename)
{
  tensorflow::MetaGraphDef meta_graph_def;
  auto status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), filename, &meta_graph_def);
  if (!status.ok())
    {
    itkGenericExceptionMacro("Can't load the input model: " << status.ToString() );
    }

  return meta_graph_def.graph_def();
}


// Get the following attributes of the specified output tensors (by name) of a graph:
// - shape
// - datatype
void GetOutputAttributes(const tensorflow::SignatureDef& signature_def, std::vector<std::string> & tensorsNames,
    std::vector<tensorflow::TensorShapeProto> & shapes, std::vector<tensorflow::DataType> & dataTypes)
{
  // Allocation
  shapes.clear();
  shapes.reserve(tensorsNames.size());
  dataTypes.clear();
  dataTypes.reserve(tensorsNames.size());

  // Get infos
  for (std::vector<std::string>::iterator nameIt = tensorsNames.begin();
      nameIt != tensorsNames.end(); ++nameIt)
  {
    bool found = false;
    std::cout << "Searching for corresponding node of  : " << (*nameIt) << std::endl;
    for (auto const & output : signature_def.outputs())
      // output is a pair (name, tensor_info)
      // cf https://stackoverflow.com/questions/63181951/how-to-get-graph-or-graphdef-from-a-given-model
    {
      if (output.first.find((*nameIt)) != std::string::npos)
        {
          found = true;
	  const tensorflow::TensorInfo& tensor_info = output.second;

	  // DEBUG
          std::cout << "\tPrintDebugString --------------------------------";
          std::cout << std::endl;
          tensor_info.PrintDebugString();
          std::cout << "\t-------------------------------------------------" << std::endl;


	  // Set default to DT_FLOAT
	  tensorflow::DataType ts_dt = tensorflow::DT_FLOAT;

	  // Default (input?) tensor type
	  ts_dt = tensor_info.dtype();
	  dataTypes.push_back(ts_dt);

	  // Get the tensor's shape
	  // Here we assure it's a tensor, with 1 shape
	  tensorflow::TensorShapeProto ts_shp = tensor_info.tensor_shape();
	  shapes.push_back(ts_shp);
      }
    }

    if (!found)
    {
      itkGenericExceptionMacro("Tensor name \"" << (*nameIt) << "\" not found" );
    }

  }

}


//
// Get the following attributes of the specified input tensors (by name) of a graph:
// - shape
// - datatype
void GetInputAttributes(const tensorflow::SignatureDef& signature_def, std::vector<std::string> & tensorsNames,
    std::vector<tensorflow::TensorShapeProto> & shapes, std::vector<tensorflow::DataType> & dataTypes)
{
  // Allocation
  shapes.clear();
  shapes.reserve(tensorsNames.size());
  dataTypes.clear();
  dataTypes.reserve(tensorsNames.size());

  
  // Get infos
  for (std::vector<std::string>::iterator nameIt = tensorsNames.begin();
      nameIt != tensorsNames.end(); ++nameIt)
  {
    bool found = false;
    std::cout << "Searching for corresponding node of  : " << (*nameIt) << std::endl;
    for (auto const & input : signature_def.inputs())
      // input is a pair (name, tensor_info)
      // cf https://stackoverflow.com/questions/63181951/how-to-get-graph-or-graphdef-from-a-given-model
    {
      if (input.first.find((*nameIt)) != std::string::npos)
        {
          found = true;
	  const tensorflow::TensorInfo& tensor_info = input.second;

	  // DEBUG
          std::cout << "\tPrintDebugString --------------------------------";
          std::cout << std::endl;
          tensor_info.PrintDebugString();
          std::cout << "\t-------------------------------------------------" << std::endl;


	  // Set default to DT_FLOAT
	  tensorflow::DataType ts_dt = tensorflow::DT_FLOAT;

	  // Default (input?) tensor type
	  ts_dt = tensor_info.dtype();
	  dataTypes.push_back(ts_dt);

	  // Get the tensor's shape
	  // Here we assure it's a tensor, with 1 shape
	  tensorflow::TensorShapeProto ts_shp = tensor_info.tensor_shape();
	  shapes.push_back(ts_shp);
      }
    }

    if (!found)
    {
      itkGenericExceptionMacro("Tensor name \"" << (*nameIt) << "\" not found" );
    }

  }

}



//
// Get the following attributes of the specified tensors (by name) of a graph:
// - shape
// - datatype
// Here we assume that the node's output is a tensor
//
void GetTensorAttributes(const tensorflow::GraphDef & graph, std::vector<std::string> & tensorsNames,
    std::vector<tensorflow::TensorShapeProto> & shapes, std::vector<tensorflow::DataType> & dataTypes)
{
  // Allocation
  shapes.clear();
  shapes.reserve(tensorsNames.size());
  dataTypes.clear();
  dataTypes.reserve(tensorsNames.size());

  // Get infos
  for (std::vector<std::string>::iterator nameIt = tensorsNames.begin();
      nameIt != tensorsNames.end(); ++nameIt)
  {
    bool found = false;
    for (int i = 0 ; i < graph.node_size() ; i++)
    {
      tensorflow::NodeDef node = graph.node(i);


      if (node.name().compare((*nameIt)) == 0)
      {
        found = true;

        // Set default to DT_FLOAT
        tensorflow::DataType ts_dt = tensorflow::DT_FLOAT;

        // Default (input?) tensor type
        auto test_is_output = node.attr().find("T");
        if (test_is_output != node.attr().end())
        {
          ts_dt = node.attr().at("T").type();
        }
        auto test_has_dtype = node.attr().find("dtype");
        if (test_has_dtype != node.attr().end())
        {
          ts_dt = node.attr().at("dtype").type();
        }
        auto test_output_type = node.attr().find("output_type");
        if (test_output_type != node.attr().end())
        {
          // if there is an output type, we take it instead of the
          // datatype of the input tensor
          ts_dt = node.attr().at("output_type").type();
        }
        dataTypes.push_back(ts_dt);

        // Get the tensor's shape
        // Here we assure it's a tensor, with 1 shape
        tensorflow::TensorShapeProto ts_shp = node.attr().at("_output_shapes").list().shape(0);
        shapes.push_back(ts_shp);
      }
    }

    if (!found)
    {
      itkGenericExceptionMacro("Tensor name \"" << (*nameIt) << "\" not found" );
    }

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
