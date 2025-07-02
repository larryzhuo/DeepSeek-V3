import sys
from pathlib import Path
# 添加父目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from cal_params import cal_mla_flops
from prefill import densmlp_flops, moe_expert_flops
from model import ModelArgs,MLA,Expert,MLP
import json
import torch
from gpu_perf import gpu


bsz = 2
seqlen = 128
num_tokens = 1
default_dtype = torch.bfloat16

def estimate_token_throughput(args, gpu_dict, seq_len, kv_cache_rate=0):
    for key in gpu_dict.keys():
        gpu = gpu_dict[key]
        mla_flops = cal_mla_flops(1, seq_len, args, kv_cache_rate)[0]  # 1 token
        mlp_flops = densmlp_flops(args, 1)
        # 假设前 n_dense_layers 层是 MLP，其余是 MoE
        n_dense = args.n_dense_layers
        n_sparse = args.n_layers - n_dense
        moe_flops = moe_expert_flops(args, 1)
        total_flops_per_token = n_dense * (mla_flops + mlp_flops) + n_sparse * (mla_flops + moe_flops)
        fp8_flops = gpu.get_fp8_flops() * 1e3  # 转为 GFLOPs
        token_per_second = fp8_flops / total_flops_per_token
        print(f"{key} 理论 token 吞吐率: {token_per_second:.2f} tokens/s")


def main(
) -> None:
    with open('../configs/config_671B.json', 'r', encoding='utf-8') as f:
        with torch.device("cuda"):
            # 创建模型前确保默认类型
            torch.set_default_dtype(default_dtype)
            # print(f"Loading config from {f.read()}")
            args = ModelArgs(**json.load(f))

            seq_len = 4383
            kv_cache_rate = 0.563
            estimate_token_throughput(args, gpu, seq_len, kv_cache_rate)


if __name__ == "__main__":
    main()