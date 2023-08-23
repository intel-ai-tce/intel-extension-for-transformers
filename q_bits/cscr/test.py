import torch
import inspect
from functools import wraps
torch.ops.load_library("build/libweight_only_jblasop.so")


def capture_args(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_strs = []
        for name, value in bound_args.arguments.items():
            arg_strs.append(f'{name}={value}')
        result = ', '.join(arg_strs)
        print(result)
        return f(*args, **kwargs)
    return wrapper


@capture_args
def test_fp32in_fp32_out(m, n, k, blocksize, compute_type, weight_type, transpose, add_bias, dump_tensor_info=False):
    activation = torch.rand(m, k, dtype=torch.float)
    wei_row=k
    wei_col=n
    if transpose:
        wei_row,wei_col=wei_col,wei_row; 
    raw_wei = torch.rand(wei_row, wei_col, dtype=torch.float)
    if dump_tensor_info:
        print(raw_wei)
    compress_wei = torch.ops.weight_only_jblasop.qbits_quantize(
        raw_wei, transpose, blocksize, compute_type, weight_type)
    revert_wei = torch.zeros(wei_row, wei_col, dtype=torch.float)
    torch.ops.weight_only_jblasop.qbits_dequantize(
        compress_wei, revert_wei, transpose, compute_type, weight_type)
    bias = torch.rand(n, dtype=torch.float)*10
    if dump_tensor_info:
        print(revert_wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    if transpose:
        revert_wei=torch.transpose(revert_wei,0,1)
    ref_dst = torch.matmul(activation, revert_wei)
    if add_bias:
        torch.ops.weight_only_jblasop.qbits_f32in_f32out_linear_with_bias(
            activation, compress_wei, bias, tar_dst, k, n, compute_type, weight_type)
    else:
        torch.ops.weight_only_jblasop.qbits_f32in_f32out_linear_without_bias(
            activation, compress_wei, tar_dst, n, k, n, compute_type, weight_type)
    if add_bias:
        ref_dst+=bias
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst)
    if torch.allclose(tar_dst,ref_dst,rtol=0.03):
        print("ok")
    else:
        print("fail")


test_fp32in_fp32_out(255, 1023, 511, 128, "int8", "s8_scalef32",True, False)# kblock must align with 128 when compute_type==int8

test_fp32in_fp32_out(255, 1023, 511, 68, "fp32",
                     "s4clip_scalef32",True, False)
test_fp32in_fp32_out(255, 1023, 511, 68, "bf16",
                     "s4clip_scalef32",True, False)
test_fp32in_fp32_out(255, 1023, 511, 128, "int8",
                     "s4clip_scalef32",True, False)# kblock must align with 128 when compute_type==int8
test_fp32in_fp32_out(255, 1023, 511, 68, "fp32",
                     "nf4_scalef32",True, False)
                     
test_fp32in_fp32_out(255, 1023, 511, 68, "bf16",
                     "nf4_scalef32",True, False)
