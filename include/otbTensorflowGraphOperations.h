/*=========================================================================

  Copyright (c) 2018-2019 Remi Cresson (IRSTEA)
  Copyright (c) 2020-2021 Remi Cresson (INRAE)


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWGRAPHOPERATIONS_H_
#define MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWGRAPHOPERATIONS_H_

// Tensorflow graph protobuf
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <tensorflow/core/protobuf/meta_graph.pb.h>

// Tensorflow SavedModel
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

// ITK exception
#include "itkMacro.h"

namespace otb {
namespace tf {

// Load SavedModel variables
void RestoreModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle);

// Save SavedModel variables
void SaveModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle);

// Load SavedModel
void LoadModel(const tensorflow::tstring path, tensorflow::SavedModelBundle & bundle, std::vector<std::string> tagList);

// Get the following attributes of the specified tensors (by name) of a graph:
// - shape
// - datatype
// Here we assume that the node's output is a tensor
void GetTensorAttributes(const tensorflow::protobuf::Map<std::string, tensorflow::TensorInfo> layers, std::vector<std::string> & tensorsNames,
    std::vector<tensorflow::TensorShapeProto> & shapes, std::vector<tensorflow::DataType> & dataTypes);

// Print a lot of stuff about the specified nodes of the graph
void PrintNodeAttributes(const tensorflow::GraphDef & graph, std::vector<std::string> & nodesNames);

} // end namespace tf
} // end namespace otb

#include "otbTensorflowGraphOperations.cxx"

#endif /* MODULES_REMOTE_OTBTENSOFLOW_INCLUDE_OTBTENSORFLOWGRAPHOPERATIONS_H_ */
