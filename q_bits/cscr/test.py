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
    raw_wei = torch.rand(k, n, dtype=torch.float)
    if dump_tensor_info:
        print(raw_wei)
    compress_wei = torch.ops.weight_only_jblasop.qbits_quantize(
        raw_wei, transpose, blocksize, compute_type, weight_type)
    revert_wei = torch.zeros(k, n, dtype=torch.float)
    torch.ops.weight_only_jblasop.qbits_dequantize(
        compress_wei, revert_wei, transpose, compute_type, weight_type)
    bias = torch.rand(n, dtype=torch.float)
    bias *= 10
    if dump_tensor_info:
        print(revert_wei)
    tar_dst = torch.zeros(m, n, dtype=torch.float)
    ref_dst = torch.matmul(activation, revert_wei)
    torch.ops.weight_only_jblasop.qbits_f32in_f32out_linear_with_bias(
        activation, compress_wei, bias, tar_dst, k, n, compute_type, weight_type)
    if dump_tensor_info:
        print(tar_dst)
        print(ref_dst+bias)


test_fp32in_fp32_out(256, 256, 256, 64, "int8",
                     "s4clip_scalef32", False, True, True)
