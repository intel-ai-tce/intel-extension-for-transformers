import torch
torch.ops.load_library("build/libweight_only_jblasop.so")
raw_wei=torch.rand(256,256,dtype=torch.float)
torch.ops.weight_only_jblasop.jblas_quantize(raw_wei,False,32,"int8","s8_scalef32")