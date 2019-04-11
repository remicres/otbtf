# -*- coding: utf-8 -*-
#==========================================================================
#
#   Copyright Remi Cresson (IRSTEA)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#==========================================================================*/
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from tricks import CheckpointToSavedModel

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",    help="checkpoint file prefix",   required=True)
parser.add_argument("--inputs",  help="input placeholder names",  required=True, nargs='+')
parser.add_argument("--outputs", help="output placeholder names", required=True, nargs='+')
parser.add_argument("--model",   help="output SavedModel",        required=True)
parser.add_argument('--clear_devices', dest='clear_devices', action='store_true')
parser.set_defaults(clear_devices=False)
params = parser.parse_args()

if __name__ == "__main__":

  CheckpointToSavedModel(params.ckpt, params.inputs, params.outputs, params.model, params.clear_devices)
  
  quit()
