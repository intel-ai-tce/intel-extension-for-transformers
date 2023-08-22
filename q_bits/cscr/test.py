import torch
torch.ops.load_library("build/libweight_only_jblasop.so")
raw_wei=torch.rand(256,256,dtype=torch.float)
print(raw_wei)
quant_wei=torch.ops.weight_only_jblasop.qbits_quantize(raw_wei,False,32,"int8","s8_scalef32")
dequant_wei=torch.zeros(256,256,dtype=torch.float)
torch.ops.weight_only_jblasop.qbits_dequantize(quant_wei,dequant_wei,False,"int8","s8_scalef32")
print(dequant_wei)