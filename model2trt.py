import torch
from torch.autograd import Variable
import tensorrt as trt
from tensorrt.parsers import onnxparser
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from trt_engine import trt_engine
import os
import numpy as np
from argparse import ArgumentParser
from resnet import resnet50
from image_reader import read_image_chw
import calib as calibrator
import time
args = ArgumentParser().parse_args()
args.input_size = 224
args.input_channel = 3
args.fc_num = 1000
args.batch_size = 4
args.onnx_model_name = "resnet50.onnx"
args.trt32_model_name = "resnet50_32.trt"
args.trt8_model_name = "resnet50_8.trt"
args.testdir='data'


def do_test(engine):
    classes=os.listdir(args.testdir)
    total = 0
    correct = 0
    start=time.time()
    for i in range(len(classes)):
        images=os.listdir(os.path.join(args.testdir,classes[i]))
        for image in images:
            gt=int(i)
            img = read_image_chw(os.path.join(args.testdir,classes[i],image),
                    args.input_size, args.input_size)
            output = engine.infer(img)
            #pdb.set_trace()
            conf, pred = torch.Tensor(np.squeeze(output[0])).topk(1, dim=0)
            pred = int(pred.data[0])
            print('gt:',classes[i],'pred:',classes[pred])
            if pred == gt:
                correct += 1
            total += 1
    end=time.time()
    return correct, total, end-start
def onnx_2_float32():
    apex = onnxparser.create_onnxconfig()
    apex.set_model_file_name(args.onnx_model_name)
    apex.set_model_dtype(trt.infer.DataType.FLOAT)
    apex.set_print_layer_info(False)
    trt_parser = onnxparser.create_onnxparser(apex)

    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)
    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(16)
    engine = builder.build_cuda_engine(trt_network)
    modelstream = engine.serialize()
    trt.utils.write_engine_to_file(args.trt32_model_name, modelstream)
    engine.destroy()
    builder.destroy()
    #engine = trt_engine('resnet',args.trt32_model_name).build_engine()

    # print ("Start Float32 Test...")
    # correct, total, use_time = do_test(engine)
    # print ('total images:',total,',time:',use_time,"s,Acc: {}".format(correct / float(total)))
def onnx_2_int8():
    apex = onnxparser.create_onnxconfig()

    apex.set_model_file_name(args.onnx_model_name)
    apex.set_model_dtype(trt.infer.DataType.FLOAT)
    apex.set_print_layer_info(False)
    trt_parser = onnxparser.create_onnxparser(apex)
    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)

    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()

    # calibration_files = create_calibration_dataset()
    batchstream = calibrator.ImageBatchStream(args)
    int8_calibrator = calibrator.PythonEntropyCalibrator(["data"], batchstream)

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

    builder = trt.infer.create_infer_builder(G_LOGGER)
    builder.set_max_batch_size(64)
    builder.set_max_workspace_size(1 << 20)
    builder.set_int8_calibrator(int8_calibrator)
    builder.set_int8_mode(True)
    engine = builder.build_cuda_engine(trt_network)
    modelstream = engine.serialize()
    trt.utils.write_engine_to_file(args.trt8_model_name, modelstream)
    engine.destroy()
    builder.destroy()
    # engine = trt_engine('resnet',args.trt8_model_name).build_engine()
    # print ("Start INT8 Test...")
    # correct, total, use_time = do_test(engine)
    # print ('total images:',total,',time:',use_time,"s,Acc: {}".format(correct / float(total)))
if __name__ == '__main__':
    if not os.path.exists(args.onnx_model_name):
        model=resnet50(pretrained=True)
        model.cuda()

        # Translate Pytorch Model into Onnx Model
        dummy_input = Variable(torch.randn(args.batch_size, args.input_channel, \
                args.input_size, args.input_size, device='cuda'))
        output_names = ["output"]
        torch.onnx.export(model, dummy_input, args.onnx_model_name, verbose=False,
                          output_names=output_names)

    onnx_2_float32()
    onnx_2_int8()
    # engine = trt_engine('resnet',args.trt32_model_name).build_engine()
    # print ("Start INT8 Test...")
    # correct, total, use_time = do_test(engine)
    # print ('total images:',total,',time:',use_time,"s,Acc: {}".format(correct / float(total)))
    # engine = trt_engine('resnet',args.trt8_model_name).build_engine()
    # print ("Start INT8 Test...")
    # correct, total, use_time = do_test(engine)
    # print ('total images:',total,',time:',use_time,"s,Acc: {}".format(correct / float(total)))
