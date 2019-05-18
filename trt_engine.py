import tensorrt as trt
from engine import Engine
class trt_engine(object):
    def __init__(self, name, trt_file, dev_id=0):
        self.dev_id = dev_id
        self.trt_file = trt_file

    def build_engine(self):
          engine_lite = Engine(device=self.dev_id,
                               PLAN=self.trt_file,
                               logger_severity=trt.infer.LogSeverity.ERROR)
          return engine_lite

