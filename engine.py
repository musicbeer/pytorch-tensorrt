#
# Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import division

import numpy as np
import tensorrt as trt
from tensorrt.parsers import uffparser, caffeparser

class Engine(object):
    '''A TensorRT engine with self containted logger, runtime, context and memory

    Members:
        - **logger** ``tensorrt.infer.Logger``: Engine Logger
        - **log_sev** ``tensorrt.infer.LogSeverity``: Verboseness of the logger
        - **max_batch_size** ``int``: Maximum supported batch size
        - **max_workspace_size** ``int``: Maximum workspace size
        - **data_type** ``tensorrt.infer.DataType``: Operating data type of the engine
        - **src_framework** ``str``: Parser used to create engine
        - **runtime** ``tensorrt.infer.Runtime``: Engine runtime
        - **engine** ``tensorrt.infer.CudaEngine``: TensorRT Engine
        - **context** ``tensorrt.infer.ExecutionContext``: Engine execution context
        - **profiler** ``tensorrt.infer.Profiler``: Engine profiler
        - **input_dim** ``[tensorrt.infer.DimsCHW]``: Input layer dimensions
        - **output_dim** ``[tensorrt.infer.DimsCHW]``: Output layer dimensions
        - **input_names** ``[str]``: Input layer names
        - **output_names** ``[str]``: Output layer names
        - **d_input** ``[pycuda.DeviceAllocation]``: GPU Buffer allocations
        - **d_output** ``[pycuda.DeviceAllocation]``: GPU Buffer allocations
        - **preprocessors** ``{str:function}``: Dictionary of input layer names and preprocessing functions (or None)
        - **postprocessors** ``{str:function}``: Dictionary of output layer names and postprocessing functions (or None)
        - **bindings** ``[int]``: Buffer Pointers

    Methods:
        - __init__(self, **kwargs)
        - __del__(self)
        - _create_engine(self, **kwargs)
        - infer(self, *input_data)
        - save(self, str)
        - supported_data_format(self, *input_data)
        - uniform_data_format(self, *input_data)
        - convert_LCHW_to_LNCHW(self, input_data)
        - transform_to_LNCHW(self, *input_data)
        - verify_data_type(self, *input_data)
        - format_data(self, output_data)
        - apply_postprocessing(self, layer_data, layer_postprocessor)
        - log_info(self, msg)
        - log_error(self, msg)
        - log_warn(self, msg)
    '''
    def __init__(self, **kwargs):
        '''Create a self contained inference engine

        Args:
            - **device** ``int`` *optional*: Device number or PCI bus ID of GPU to use *Default: ``0``*
            - **logger** ``<class> tensorrt.infer.Logger`` *optional*: Engine Logger *Default: ``tensorrt.infer.ColorLogger``*
            - **log_sev** ``tensorrt.infer.LogSeverity`` *optional*: Logger Verboseness *Default: ``tensorrt.infer.LogSeverity.INFO``*
            - **max_batch_size** ``int`` *optional*: Max batch size *Default: ``1``*
            - **max_workspace_size** ``int`` *optional*: Max worspace size *Default: ``1 << 20``*
            - **data_type** ``tensorrt.infer.DataType`` *optional*: Operating Data Type *Default: ``tensorrt.infer.DataType.FLOAT``*
            - **framework** ``str``: Source framework ("tf", "uff", "caffe")
                - "tf" and "uff" requires one of the following arguments:
                    - **path** ``str``: Path to frozen file
                    - **stream** ``str/bytes``: Serialized graph
                - "caffe" requires both a path to deploy file and path to model file
                    - **deployfile** ``str``: Path to deploy file
                    - **modelfile** ``str``: Path to model file
                and
                - **input_nodes** ``{str : (C, H, W)/[C, H, W]}``: Names and sizes of input nodes
                - **output_nodes** ``[str]``: Names of output nodes
            - **PLAN** ``str``: Path to a PLAN file
            - **engine_stream** ``str/[bytes]``: serialized TensorRT engine stream
            - **calibrator** ``tensorrt.infer.Calibrator`` *optional*: Int8 Calibrator
            - **profiler** ``<class> tensorrt.infer.Profiler`` *optional*: Engine profiler
            - **preprocessors** ``{str:function}``: Dictionary of input layer names and preprocessing functions (or None)
            - **postprocessors** ``{str:function}``: Dictionary of output layer names and postprocessing functions (or None)

        '''
        # We only need to import these if the Engine is actually created.
        try:
            from importlib import import_module
            self.cuda = import_module("pycuda.driver")
            self.cuda.init()
        except ImportError as err:
            raise ImportError("""ERROR: Failed to import module({})
Please make sure you have pycuda and the example dependencies installed.
sudo apt-get install python(3)-pycuda
pip install tensorrt[examples]""".format(err))

        device = 0 if not "device" in kwargs else kwargs["device"]
        self.device = self.cuda.Device(device)
        self.cuda_context = self.device.make_context()
        # Push the context - and make sure to pop it before returning!
        self.cuda_context.push()

        logger = trt.infer.ColorLogger
        log_sev = trt.infer.LogSeverity.INFO
        self.max_batch_size = 1
        self.max_workspace_size = 1 << 20
        self.data_type = trt.infer.DataType.FLOAT
        # Do all CUDA allocations with this size. If you try to allocate with HALF, it will produce incorrect results.
        self.alloc_data_type = trt.infer.DataType.FLOAT

        self.src_framework = None
        self.context = None
        self.engine = None
        self.runtime = None

        for k,v in kwargs.items():
            if k == "logger":
                logger = v
            elif k == "logger_severity":
                log_sev = v
            elif k == "max_batch_size":
                self.max_batch_size = v
            elif k == "max_workspace_size":
                self.max_workspace_size = v
            elif k == "data_type":
                self.data_type = v

        self.logger = logger(log_sev)
        self.runtime = trt.infer.create_infer_runtime(self.logger)

        frwk = kwargs.get("framework", None)
        if frwk:
            self.log_info("Detecting Framework")
            modelstream = None
            if frwk == "tf" or frwk == "tensorflow":
                self.src_framework = "uff"
                path = kwargs.get("path", None)
                stream = kwargs.get("stream", None)
                output_nodes = kwargs.get("output_nodes", None)
                input_nodes = kwargs.get("input_nodes", None)
                if not (bool(path) ^ bool(stream)) or not output_nodes or not input_nodes:
                    self.log_error("Need either modelstream or filepath, output nodes and input nodes to create engine from tensorflow graph")

                try:
                    import uff
                except ImportError as err:
                    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the UFF toolkit installed.
For installation instructions, see:
https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/#python and click on the 'TensoRT Python API' link""".format(err))

                if stream:
                    modelstream = uff.from_tensorflow(stream, output_nodes)
                else:
                    modelstream = uff.from_tensorflow_frozen_model(path, output_nodes)

            elif frwk == "uff":
                self.src_framework = "uff"
                path = kwargs.get("path", None)
                stream = kwargs.get("stream", None)
                output_nodes = kwargs.get("output_nodes", None)
                input_nodes = kwargs.get("input_nodes", None)
                if not (bool(path) ^ bool(stream)) or not output_nodes  or not input_nodes:
                    self.log_error("Need either modelstream or filepath, input nodes and output nodes to create engine from uff graph")
                if stream:
                    modelstream = stream
            elif frwk == "caffe" or frwk == "c1":
                self.src_framework = "caffe"
                modelfile = kwargs.get("modelfile", None)
                deployfile = kwargs.get("deployfile", None)
                output_nodes = kwargs.get("output_nodes", None)
                if not (modelfile and deployfile) or not output_nodes:
                    self.log_error("Need a model file and deployfile, and output nodes to create engine from caffe")
            else:
                self.log_error("Unsupported framework")

            self._create_engine(modelstream, **kwargs)

        elif kwargs.get("PLAN", None):
            self.engine = trt.utils.load_engine(self.logger, kwargs.get("PLAN", None), kwargs.get("plugins", None))
            self.max_batch_size = self.engine.get_max_batch_size()
        elif kwargs.get("engine_stream", None):
            self.engine = self.runtime.deserialize_cuda_engine(kwargs.get("engine_stream", None), kwargs.get("plugins", None))
            self.max_batch_size = self.engine.get_max_batch_size()
        else:
            self.log_error("No supported engine source provided")

        self.log_info("Verifying engine construction was successful")
        assert(self.engine)

        self.context = self.engine.create_execution_context()

        if kwargs.get("profiler", None):
            self.profiler = kwargs.get("profiler", None)
            self.context.set_profiler(self.profiler)

        self.log_info("Allocating GPU buffers")
        nb_bindings = self.engine.get_nb_bindings()

        input_index = []
        output_index = []
        for b in range(nb_bindings):
            if self.engine.binding_is_input(b):
                input_index.append(b)
            else:
                output_index.append(b)

        self.input_dim = [self.engine.get_binding_dimensions(i).to_DimsCHW() for i in input_index]
        self.output_dim = [self.engine.get_binding_dimensions(o).to_DimsCHW() for o in output_index]

        insize = [self.max_batch_size * i.vol() * self.alloc_data_type.input_type().itemsize for i in self.input_dim]
        outsize = [self.max_batch_size * o.vol() * self.alloc_data_type.input_type().itemsize for o in self.output_dim]

        self.d_input = [self.cuda.mem_alloc(i) for i in insize]
        self.d_output = [self.cuda.mem_alloc(o) for o in outsize]

        self.bindings = [int(i) for i in self.d_input]
        self.bindings = self.bindings + [int(i) for i in self.d_output]

        self.input_names = [self.engine.get_binding_name(i) for i in input_index]
        self.output_names = [self.engine.get_binding_name(o) for o in output_index]

        self.has_preprocessors, self.has_postprocessors = False, False
        if kwargs.get("preprocessors", None):
            if len(kwargs["preprocessors"]) < len(self.input_dim):
                self.log_error('''In order to use the internal preprocessor, a function or a None must be provided for each input node, saw {}, expected {}'''.format(len(kwargs["preprocessors"]),len(self.input_dim)))
            elif len(kwargs["preprocessors"]) > len(self.input_dim):
                 self.log_error('''Too many functions in preprocessing function table, saw {}, expected {}'''.format(len(kwargs["preprocessors"]),len(self.input_dim)))
            else:
                self.log_info("Registering preprocessor function table")
                self.preprocessors = kwargs["preprocessors"]
                self.has_preprocessors = True

        if kwargs.get("postprocessors", None):
            if len(kwargs["postprocessors"]) < len(self.input_dim):
                self.log_error('''In order to use the internal postprocessor, a function or a None must be provided for each output node, saw {}, expected {}'''.format(len(kwargs["postprocessors"]),len(self.input_dim)))

            elif len(kwargs["postprocessors"]) > len(self.input_dim):
                self.log_error('''Too many functions in postprocessor function table, saw {}, expected {}'''.format(len(kwargs["postprocessors"]),len(self.input_dim)))

            else:
                self.log_info("Registering postprocessor function table")
                self.postprocessors = kwargs["postprocessors"]
                self.has_postprocessors = True

        # Remove this from the context stack so Lite Engine is self contained.
        self.cuda_context.pop()
        self.runtime.destroy()
        self.runtime = None

    def __del__(self):
        if self.cuda_context:
            # Must remove this context on destruction.
            self.cuda_context.pop()
        if self.context:
            self.context.destroy()
        if self.engine:
            self.engine.destroy()
        if self.runtime:
            self.runtime.destroy()

    def _create_engine(self, modelstream, **kwargs):
        '''
        Helper to create engine when trying to build from models
        '''
        self.log_info("Parsing Model from {}".format(self.src_framework))
        if self.src_framework == "uff":
            parser = uffparser.create_uff_parser()
            for k,v in kwargs["input_nodes"].items():
                parser.register_input(k,v, 0)

            for o in kwargs["output_nodes"]:
                parser.register_output(o)

            if modelstream:
                self.engine = trt.utils.uff_to_trt_engine(self.logger,
                                                          modelstream,
                                                          parser,
                                                          self.max_batch_size,
                                                          self.max_workspace_size,
                                                          self.data_type,
                                                          None, #TODO: Figure out if plugins are supported in UFF
                                                          kwargs.get("calibrator", None))
            else:
                self.engine = trt.utils.uff_file_to_trt_engine(self.logger,
                                                               kwargs["path"],
                                                               parser,
                                                               self.max_batch_size,
                                                               self.max_workspace_size,
                                                               self.data_type,
                                                               None, #TODO: Figure out if plugins are supported in UFF
                                                               kwargs.get("calibrator", None))

            parser.destroy()

        elif self.src_framework == "caffe":
            self.engine = trt.utils.caffe_to_trt_engine(self.logger,
                                                        kwargs["deployfile"],
                                                        kwargs["modelfile"],
                                                        self.max_batch_size,
                                                        self.max_workspace_size,
                                                        kwargs["output_nodes"],
                                                        self.data_type,
                                                        kwargs.get("plugins", None),
                                                        kwargs.get("calibrator", None))

    def save(self, path):
        '''
        Save the TensorRT Engine to a PLAN file

        Saves the TensorRT Engine to a PLAN file that can be used later.
        *Note:* This saves the only the internal TensorRT engine in the class
        not the Engine object and all provided settings.

        Args:
            - **path** ``str``: Desired path to save file
        '''
        trt.utils.write_engine_to_file(path, self.engine.serialize())

    def supported_data_format(self, *input_data):
        '''
        Dectects wether the provided data is one of the supported types

        Args:
            - **input_data** ``tuple`` Tuple of lists of data for input layers

        Returns:
            - ``None``

        Side-Effects:
            - Sets the input format detected (LLCHW, LNCHW, LCHW, ZNCHW, NCHW, CHW)

        Raises:
            - ``Value Error``: Unsupported data format
        '''
        self.log_info("Detecting input data format")
        if type(input_data[0]) is list: #Num Batches?, Batches?
            if type(input_data[0][0]) is list: #Batches?
                if type(input_data[0][0][0]) is np.ndarray and input_data[0][0][0].ndim == 3:
                    self.input_format = "LLCHW"
                else:
                    self.log_error('''Unsupported data format, expect 5D numpy, List of NCHWs, List of CHWs, List of List of CHWs, Single NCHW or Single CHW''')

            elif type(input_data[0][0]) is np.ndarray: #Batches?, Img?
                if input_data[0][0].ndim == 4: #List of NCHW
                    self.input_format = "LNCHW"
                elif input_data[0][0].ndim == 3: #List of CHW
                    self.input_format = "LCHW"
                else:
                    self.log_error('''Unsupported data format, expect 5D numpy, List of NCHWs, List of CHWs, List of List of CHWs, Single NCHW or Single CHW''')

        elif type(input_data[0]) is np.ndarray:
            if input_data[0].ndim == 5:
                self.input_format = "ZNCHW"
            elif input_data[0].ndim == 4:
                self.input_format = "NCHW"
            elif input_data[0].ndim == 3:
                self.input_format = "CHW"
            else:
                self.log_error('''Unsupported data format, expect 5D numpy, List of NCHWs, List of CHWs, List of List of CHWs, Single NCHW or Single CHW''')

        else:
             self.log_error('''Unsupported data format, expect 5D numpy, List of NCHWs, List of CHWs, List of List of CHWs Single NCHW or Single CHW''')

        self.log_info("Dectected data format {}".format(self.input_format))

    def uniform_data_format(self, *input_data):
        '''
        Verifies that the data format is uniform accross all input layers and
        that it is uniform accross batches

        Args:
            - **input_data** ``tuple`` Tuple of lists of data for input layers

        Returns:
            - ``None``

        Raises:
            - ``Value Error``: If number of batches, batch size or CHW dims differ or if the batch size is too big
        '''
        self.log_info("Verifying data format is uniform accross all input layers")
        for i in input_data:
            #CHECK NUM BATCHES IS THE SAME AND BATCH SIZES MATCH
            if self.input_format == "LLCHW":
                #NUM BATCHES
                if len(input_data[0]) != len(i):
                    self.log_error('''Input format detected as [[CHW]], len([[CHW]]) for input layers not uniform a.k.a number of batches not uniform, expected {}, saw {}'''.format(len(input_data[0]),len(i)))

                #BATCH SIZES MATCH
                for j in range(len(input_data[0])):
                    if len(input_data[0][j]) != len(i[j]):
                        self.log_error('''Input format detected as [[CHW]], len([CHW]) for input layers not uniform a.k.a batch size not uniform, expected {}, saw {}'''.format(len(input_data[0][j]),len(i[j])))

                    if len(i[j]) > self.max_batch_size:
                        self.log_error('''Input format detected as [[CHW]], len([CHW]) greater than engine max batch size, saw {}, max {}'''.format(len(i[j]), self.max_batch_size))
                #CHECK SAME CHW DIMS
                dims = i[0][0].shape
                for j in i:
                    for k in j:
                        if k.shape != dims:
                            self.log_error('''Input format detected as [[CHW]], found CHW tensors to not be uniform, saw {}, expected {}'''.format(k.shape, dims))

            #CHECK NUM BATCHES IS THE SAME AND BATCH SIZES MATCH
            elif self.input_format == "LNCHW":
                #NUM BATCHES
                if len(input_data[0]) != len(i):
                    self.log_error('''Input format detected as [NCHW], len([NCHW]) for input layers not uniform a.k.a number of batches not uniform, expected {}, saw {}'''.format(len(input_data[0]),len(i)))
                for j in range(len(input_data[0])):
                    if input_data[0][j].shape[0] != i[j].shape[0]:
                        self.log_error('''Input format detected as [NCHW], batch size not uniform, expected {}, saw {}'''.format(input_data[0][j].shape[0], i[j].shape[0]))
                dims = i[0].shape
                for j in i:
                    if j.shape != dims:
                        self.log_error('''Input format detected as [NCHW], found NCHW tensors to not be uniform in the list, saw {}, expected {}'''.format(j.shape, dims))

            #CHECK BATCH SIZES MATCH
            elif self.input_format == "LCHW":
                if len(input_data[0]) != len(i):
                    self.log_error('''Input format detected as [CHW], len([CHW]) for input layers
 not uniform a.k.a batch sizes not uniform, expected {}, saw {}'''.format(len(input_data[0]),len(i)))
                dims = i[0].shape
                for j in i:
                    if j.shape != dims:
                        self.log_error('''Input format detected as [CHW],
found CHW tensors to not be uniform, saw {}, expected {}'''.format(j.shape, dims))

            #CHECK NUM BATCHES ARE THE SAME AND BATCH SIZES MATCH
            elif self.input_format == "ZNCHW":
                if input_data[0].shape[0] != i.shape[0]:
                    self.log_error('''Input format detected as ZNCHW,
for input layes number of batches not uniform, expected {}, saw {}'''.format(input_data[0].shape[0],i.shape[0]))
                if input_data[0].shape[1] != i.shape[1]:
                    self.log_error('''Input format detected as ZNCHW,
for input layes batch sizes not uniform, expected {}, saw {}'''.format(len(input_data[0].shape[1]),i.shape[1]))
            #CHECK BATCHES ARE THE SAME
            elif self.input_format == "NCHW":
                if input_data[0].shape[0] != i.shape[0]:
                    self.log_error('''Input format detected as NCHW,
for input layes batch sizes not uniform, expected {}, saw {}'''.format(len(input_data[0].shape[0]),i.shape[0]))
            elif self.input_format == "CHW":
                pass
            else:
                self.log_error("Engine input format set to something other than supported types, saw {}".format(self.input_format))

    def convert_LCHW_to_LNCHW(self, input_data):
        '''
        Converts data from LCHW format to LNCHW

        Helper for transform_to_LNCHW to convert LCHW format to LNCHW

        Args:
            - **input_data** ``list`` List of lists of LCHW data

        Returns:
            - ``list`` List of list of LNCHW data
        '''
        for i in range(len(input_data)):
            data = []
            if len(input_data[i]) > self.max_batch_size:
                for b in range(len(input_data[i]) // self.max_batch_size):
                    data.append(np.array(input_data[i][(b * self.max_batch_size) : ((b+1) * self.max_batch_size)]))
                if len(input_data[i]) % self.max_batch_size:
                    data.append(np.array(input_data[i][-(len(input_data[i]) % self.max_batch_size):]))
            else:
                data.append(np.array(input_data[i]))
            input_data[i] = data
        return input_data

    def transform_to_LNCHW(self, *input_data):
        '''
        Converts supported data formats to LNCHW

        Converts data from LLCHW, LCHW, ZNCHW, NCHW, CHW to LNCHW (List of batches)

        For LLCHW, ZNCHW, LNCHW; batch size (Length of internal lists) must be smaller than
        the engine's max batch size (default: 1)

        Args:
            - **input_data** ``tuple`` Tuple of lists of data for input layers

        Returns:
            - ``list``: Data formated as a list of batches for inference

        Raises:
            - ``Value Error``: Batch size too large
        '''
        if self.input_format == "LLCHW":
            input_data_list = []
            for i in range(len(input_data)):
                data = []
                for j in range(len(input_data[i])):
                    if len(input_data[i][j]) > self.max_batch_size:
                        self.log_error("Batch size is too large for engine, saw {}, max: {}".format(i[0].shape[0],
                                                                                                self.max_batch_size))
                    data.append(np.array(input_data[i][j]))
                input_data_list.append(data)
            return input_data_list


        elif self.input_format == "LNCHW":
            for i in input_data:
                for b in i:
                    if b.shape[0] > self.max_batch_size:
                        self.log_error("Batch size is too large for engine, saw {}, max: {}".format(i[0].shape[0],
                                                                                                self.max_batch_size))
            return input_data

        elif self.input_format == "LCHW":
            input_data_list = []
            for i in input_data:
                input_data_list.append(i)
            return self.convert_LCHW_to_LNCHW(input_data_list)

        elif self.input_format == "ZNCHW":
            input_data_list = []
            for i in input_data:
                if i[0].shape[1] > self.max_batch_size:
                    self.log_error("Batch size is too large for engine, saw {}, max: {}".format(i[0].shape[0],
                                                                                                self.max_batch_size))
            for i in range(len(input_data)):
                data = []
                for b in range(len(input_data[i])):
                    data.append(input_data[i][b])
                input_data_list.append(data)
            return input_data_list

        elif self.input_format == "NCHW":
            needs_reshaping = False
            input_data_list = []
            for i in input_data:
                if i.shape[0] > self.max_batch_size:
                    self.log_warn('''Batch size is too large for engine, saw {}, max: {}, trying internal reshaping'''.format(i.shape[0], self.max_batch_size))
                    needs_reshaping = True
                    break

            if needs_reshaping:
                for i in range(len(input_data)):
                    data = input_data[i]
                    batches = []
                    for b in range(input_data[i].shape[0] // self.max_batch_size):
                        batches.append(data[(b * self.max_batch_size) : ((b + 1) * self.max_batch_size)])
                    if input_data[i].shape[0] % self.max_batch_size:
                        batches.append(data[-(input_data[i].shape[0] % self.max_batch_size):])
                    input_data_list.append(batches)
            else:
                for i in input_data:
                    input_data_list.append([i])

            return input_data_list

        elif self.input_format == "CHW":
            input_data_list  = []
            for i in range(len(input_data)):
                data = []
                data.append(input_data[i])
                input_data_list.append([np.array(data)])
            return input_data_list

    def verify_data_type(self, input_data):
        '''
        Verifies if the data type is the expected type for the engine

        Verifies if the data type is the expected type for the engine,
        will attempt to convert the data to the expected type.

        Will provide a warning if a different data type is detected.
        May be the cause of incorrect results from the engine, if the
        data cannot be converted.

        Args:
            - **input_data** ``list``: List of LNCHW data

        Returns:
            - ``list``: List of LNCHW data with the correct type

        Side-Effects:
           - If data type is changed, warning will be printed in the engine logger,
             make sure logger severity is to at least warning to see.

        '''
        self.log_info("Verifying batches are the expected data type")
        for l in input_data:
            for b in range(len(l)):
                if l[b].dtype != self.data_type.input_type():
                    self.log_warn("Input batch data type is not the expected data type for the engine, saw {}, expected {}. Attempting conversion".format(l[b].dtype, self.data_type.input_type()))
                    l[b] = l[b].astype(self.data_type.input_type())
        return input_data

    def infer(self, *input_data):
        '''
        Run inference on a set of data

        Runs inference on a set of data provided by the user.
        Data must be provided in a supportted data format:
            - CHW: Single 3D numpy array in form Channels x Height x Width
            - NCHW: Single 4D numpy array in form Batch Size x Channels x Height x Width *note: If the batch size is larger than the supported max batch size, the function will attempt to split up the batch into smaller supported batches*
            - ZNCHW: Single 5D numpy array in the form Number of Batces x Batch size x Channels x Height x Width *note: if the batch size is larger than the supported max batch size, the function will error out*
            - LNCHW: List of 4D numpy arrays in form Batch Size x Channels x Height x Width *note: if the batch size is larger than the supported max batch size, the function will error out*
            - LLCHW: List of lists of 3D numpy arrays in form Channels x Height x Width *note: if the size of the inner lists are is larger than the supported max batch size, or the size of the inner lists are not uniform the function will error out*
            - LCHW: List of 3D numpy arrays in form Channels x Height x Width *note: If the size of the list is larger than the supported max batch size, the function will attempt to split up the list into smaller supported batches*

        Provide a seperate array for each input layer

        If a preprocessor function table is registered with the engine at creation then before inference, each input data object (3D numpy array) will be passed into the user specified preprocessor function for the relevant input layer and inference will be run on the preprocessed data.

        If a postprocessor function table is registered with the engine at creation then before inference, each output data object (3D numpy array) will be passed into the user specified postprocessor function for the relevant output layer and the function will return the postprocessed data.

        Args:
            - input_data,... ``list/numpy.ndarray``: List or numpy array containing data in a supported format, multiple lists/np.ndarrays should be passed in if there are multiple input layers, one per layer in order of bindings

        Returns:
            - ``list/numpy.ndarray``: Results of inference arranged in the same format the input data was
        '''
        # Make this the active context - make sure to pop before returning!
        self.cuda_context.push()

        if len(input_data) != len(self.d_input):
            if len(input_data) > len(self.d_input):
                self.log_error("Too many input data streams, saw {}, expected {}".format(len(input_data),
                                                                                         len(self.d_input)))
            else:
                self.log_error("Too few input data streams, saw {}, expected {}".format(len(input_data),
                                                                                         len(self.d_input)))

        #Expect 5D Tensor, lists of NCHWs, lists of CHWs, single NCHW or single CHW
        self.supported_data_format(*input_data)
        self.uniform_data_format(*input_data)
        input_data = self.transform_to_LNCHW(*input_data)
        input_data = self.verify_data_type(input_data)

        num_execs = len(input_data[0])

        self.log_info("Executing inference")

        self.log_info("Number of Batches: {}".format(num_execs))

        output_data = []
        for o in self.output_dim:
            layer_out = []
            for b in input_data[0]:
                layer_out.append(np.empty((b.shape[0], o.C(), o.H(), o.W()), dtype=self.alloc_data_type.input_type()))
            output_data.append(layer_out)

        #CAN YOU CREATE MORE EXECUTION STREAMS?
        stream = self.cuda.Stream()
        for n in range(num_execs):
            #Transfer nth batch for each input layer
            for i in range(len(input_data)):
                #In pipeline batch element preprocessing
                if self.has_preprocessors and self.preprocessors[self.input_names[i]]:
                    old = input_data[i][n][0]
                    for chw in range(len(input_data[i][n])):
                        input_data[i][n][chw] = self.preprocessors[self.input_names[i]](input_data[i][n][chw])
                #Transfer data to device
                self.cuda.memcpy_htod_async(self.d_input[i], np.ascontiguousarray(input_data[i][n].astype(self.alloc_data_type.input_type())), stream)
            stream.synchronize()
            self.log_info("Execution batch size: {}".format(input_data[i][n].shape[0]))
            self.context.enqueue(input_data[i][n].shape[0], self.bindings, stream.handle)
            stream.synchronize()
            for o in range(len(self.d_output)):
                self.cuda.memcpy_dtoh_async(output_data[o][n], self.d_output[o], stream)
            stream.synchronize()

        #TODO: USE MULTIPLE STREAMS SO PEOPLE CAN GIVE DATA LARGER THAN BATCH SIZE (NEEDS EVENTS FROM PYCUDA)
        output_data = self.format_data(output_data)
        for o in range(len(output_data)):
            if self.has_postprocessors and self.postprocessors[self.output_names[o]]:
                self.log_info("Postprocessing output data for output layer " + self.output_names[o])
                output_data[o] = self.apply_postprocessing(output_data[o], self.postprocessors[self.output_names[o]])

        # Remove this from the context stack so infer is self contained.
        stream = None
        self.cuda_context.pop()
        return output_data

    def format_data(self, output_data):
        '''
        Format results to same shape as input data

        Format the results from inference from the operating data format (LNCHW) to the same data format that input data was given in.

        Args:
            - **output_data** ``[LNCHW data]``: List of inference results in LNCHW

        Returns:
            - ``list``: List of results for each layer in the same format as the input data
        '''
        self.log_info("Formating output data")
        frmt = self.input_format
        formatted_output_data = []
        for o in range(len(output_data)):
            if frmt == "LLCHW":
                batches = []
                for b in range(len(output_data[o])):
                    batch = []
                    for c in range(len(output_data[o][b])):
                        batch.append(output_data[o][b][c])
                    batches.append(batch)
                formatted_output_data.append(batches)

            elif frmt == "LCHW":
                batch_data = []
                for b in range(len(output_data[o])):
                    for c in range(len(output_data[o][b])):
                        batch_data.append(output_data[o][b][c])
                formatted_output_data.append(batch_data)

            elif frmt == "LNCHW":
                return output_data

            elif frmt == "ZNCHW":
                formatted_output_data.append(np.array(output_data[o]))

            elif frmt == "NCHW":
                batch_data = []
                for b in range(len(output_data[o])):
                    for c in range(len(output_data[o][b])):
                        batch_data.append(output_data[o][b][c])
                formatted_output_data.append(np.array(batch_data))

            elif frmt == "CHW":
                formatted_output_data.append(output_data[o][0][0])

        return formatted_output_data

    def apply_postprocessing(self, layer_data, layer_postprocessor):
        '''
        Apply the user specified postprocessing function to results

        Takes outputs for a layer and the layer's specified postprocessor
        function and processes each result (3D numpy array). Stores the
        result in the place of the 3D Numpy array.

        Args:
            - **layer_data** ``list/numpy.ndarray``: Formated results for a layer
            - **layer_postprocessor** ``function``: Layer's post processing function

        Results:
            - ``list/numpy.ndarray``: Post processed results

        '''
        frmt = self.input_format
        if frmt == "LLCHW":
            for i in range(len(layer_data)):
                for j in range(len(layer_data[i])):
                    layer_data[i][j] = layer_postprocessor(layer_data[i][j])
            return layer_data

        elif frmt == "LCHW":
            for i in range(len(layer_data)):
                layer_data[i] = layer_postprocessor(layer_data[i])
            return layer_data

        elif frmt == "LNCHW":
            for i in range(len(layer_data)):
                new_data = []
                for j in range(len(layer_data[i])):
                    new_data.append(layer_postprocessor(layer_data[i][j]))
                layer_data[i] = np.array(new_data)
            return layer_data

        elif frmt == "ZNCHW":
            new_data_container = []
            for i in range(len(layer_data)):
                new_data = []
                for j in range(len(layer_data[i])):
                    new_data.append(layer_postprocessor(layer_data[i][j]))
                new_data_container.append(np.array(new_data))
            return np.array(new_data_container)

        elif frmt == "NCHW":
            new_data = []
            for i in range(len(layer_data)):
                new_data.append(layer_postprocessor(layer_data[i]))
            return np.array(new_data)

        elif frmt == "CHW":
            return layer_postprocessor(layer_data)

        return layer_data

    def log_info(self, msg):
        '''
        Helper for printing engine status

        Args:
            - **msg** ``str``: What to print

        Side-effects:
            - Prints message to console in the INFO stream
        '''
        self.logger.log(trt.infer.LogSeverity.INFO, msg)

    def log_warn(self, msg):
        '''
        Helper for printing engine warnings

        Args:
            - **msg** ``str``: What to print

        Side-effects:
            - Prints message to console in the WARNING stream
        '''
        self.logger.log(trt.infer.LogSeverity.WARNING, msg)

    def log_error(self, msg):
        '''
        Helper for printing engine errors

        Args:
            - **msg** ``str``: What to print

        Side-effects:
            - Prints message to console in the ERROR stream

        Raises:
            - ValueError
        '''
        self.logger.log(trt.infer.LogSeverity.ERROR, msg)
        raise ValueError(msg)

