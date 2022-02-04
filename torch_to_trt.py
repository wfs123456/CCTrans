import torch
from torch2trt import torch2trt
from Networks import ALTGVT
import argparse

import os
import time
from torch2trt import tensorrt_converter
import tensorrt as trt
from torch2trt.torch2trt import *


def load_model(args, FP_16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    model = ALTGVT.alt_gvt_large(pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(args.weight_path, device))
    model.eval()
    return model if not FP_16 else model.half()



@tensorrt_converter('torch.Tensor.transpose')
def convert_transpose(ctx):
    input = ctx.method_args[0]
    #input = input.unsqueeze(0) # add batch
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0] 
    output = ctx.method_return
    # permutation -1 because TRT does not include batch dim
    permutation = list(range(len(input.shape) - 1))
    dim0 = ctx.method_args[1] - 1
    dim1 = ctx.method_args[2] - 1

    permutation[dim0] = dim1
    permutation[dim1] = dim0
    layer = ctx.network.add_shuffle(input_trt)

    layer.second_transpose = tuple(permutation)

    output._trt = layer.get_output(0)

#@tensorrt_converter('torch.Tensor.__matmul__')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.Tensor.softmax')
def convert_softmax(ctx):
    input = ctx.method_args[0]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return

    # get dims from args or kwargs
    if 'dim' in ctx.method_kwargs:
        dim = ctx.method_kwargs['dim']
    elif len(ctx.method_args) >= 2:
        dim = ctx.method_args[1]
        
    # convert negative dims
#     import pdb
#     pdb.set_trace()
    if dim < 0:
        dim = len(input.shape) + dim

    axes = 1 << (dim - 1)

    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = axes

    output._trt = layer.get_output(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='assign device')

    parser.add_argument("--weight-path", type=str, required=True,
                        help="the weight path to be loaded")
    args = parser.parse_args()
    print(args)



    model = load_model(args, FP_16=False)

    # create example data
    x = torch.ones((4, 3, 256, 256)).cuda()

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(model, [x], max_batch_size=226)

    x = torch.ones((4, 3, 256, 256)).cuda()
    
    start = time.time()
    y, _ = model(x)
    print("Inference (pt) in :", round(time.time() - start, 2))

    start = time.time()
    y_trt, _ = model_trt(x)
    print("Inference(trt) in :", round(time.time() - start, 2))

    # save
    torch.save(model_trt.state_dict(), 'model_weights/trt_512x512.pth')