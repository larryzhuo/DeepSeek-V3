import sys
from pathlib import Path
# 添加父目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from model import ModelArgs,MLA,Expert,MLP
import json
from argparse import ArgumentParser
from ptflops import get_model_complexity_info
import torch

bsz = 2
seqlen = 128
num_tokens = 1
default_dtype = torch.bfloat16


"""
包装 MLA，添加额外参数
"""
class WrappedMLA(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mla = MLA(args)
        # 3090 不支持 fp8,用不了 torch.float8_e4m3fn
        self.dtype = default_dtype

        # 初始化所有张量使用一致的数据类型
        self.freqs_cis = torch.randn(seqlen, args.qk_rope_head_dim//2, dtype=self.dtype)
        print(f"freqs_cis shape: {self.freqs_cis.shape}")

        self.mask = torch.full((seqlen, seqlen), float("-inf"), dtype=self.dtype)
        self.mask = torch.triu(self.mask, diagonal=1)
        
    def forward(self, x, start_pos=0):
        # 自动转换输入数据类型
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        print(f"x.shape={x.shape}")
        if x.dim() == 4:  # 如果输入是 4D，去掉第一个维度; (1, 2, 128, 7168)降成 (2, 128, 7168)
            x = x.squeeze(0)
        return self.mla(x, start_pos, self.freqs_cis, self.mask)

def cal_mla_flops(q_len, kv_len, args:ModelArgs, kv_cache_rate=0):
    """
    mla flops手工计算
    这里是/1e9 而不是 /1e12,所以返回的都是 GFLOPS
    """
    #calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank #wq_a
    q_up_proj = q_len * args.q_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.qk_rope_head_dim) #wq_b
    kv_down_proj = kv_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a
    k_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.qk_nope_head_dim #w_uk
    v_up_proj = kv_len * args.kv_lora_rank * args.n_heads * args.v_head_dim #w_uv

    kv_down_proj = kv_down_proj * (1 - kv_cache_rate)
    gemm_sum = q_down_proj + q_up_proj + kv_down_proj + k_up_proj + v_up_proj
    
    #把它看成一个标准的args.n_heads的MHA
    mha = args.n_heads * ( q_len * args.qk_rope_head_dim * kv_len #QK_score_rope
                          + q_len * args.qk_nope_head_dim * kv_len #QK_score_nope
                          + q_len * kv_len * args.v_head_dim) #ScoreV
    wo = q_len * args.n_heads * args.v_head_dim * args.dim #wo
    attn_sum = mha + wo
    
    #return flops by 2* Sum(MACs)
    GEMM_FP8_FLOPS = gemm_sum * 2/1e9
    ATTN_FP16_FLOPS =  attn_sum * 2/1e9
    
    return GEMM_FP8_FLOPS+ATTN_FP16_FLOPS, GEMM_FP8_FLOPS,ATTN_FP16_FLOPS

def mla_matabsob_flops(q_len, kv_len, args:ModelArgs, kv_cache_rate=0):
    #calculate MACs and estimate Flops approx. 2xMAC.
    q_down_proj = q_len * args.dim * args.q_lora_rank #wq_a
    q_rope_up_proj = q_len * args.q_lora_rank * args.n_heads * args.qk_rope_head_dim #wq_b_rope
    q_absorb = q_len * args.n_heads * args.q_lora_rank * args.kv_lora_rank 
    
    kv_down_proj = kv_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim) #wkv_a
    kv_down_proj = kv_down_proj * (1 - kv_cache_rate) #KV-Cache命中率修正
    gemm_sum = q_down_proj + q_rope_up_proj + q_absorb + kv_down_proj 
    
    #把它看成一个标准的args.n_heads的MQA
    mqa = args.n_heads * ( q_len * args.qk_rope_head_dim * kv_len #Score_rope
                          + q_len * args.kv_lora_rank * kv_len #Score_nope
                          + q_len * kv_len * args.kv_lora_rank) #Score V
    o_absorb = q_len * args.n_heads * args.kv_lora_rank * args.dim 
    attn_sum = mqa + o_absorb
    
    #return flops by 2* Sum(MACs)
    gemm_sum =  gemm_sum * 2/1e9
    attn_sum = attn_sum * 2/1e9
    
    return gemm_sum + attn_sum, gemm_sum,attn_sum



def cal_mla(args):
    """
    计算 mla
    """
    wrapped_mla = WrappedMLA(args)
    # 测试forward
    test_input = torch.randn(bsz, seqlen, args.dim, dtype=default_dtype)
    output = wrapped_mla(test_input)
    print(f"Output shape: {output.shape}")  # 验证输出形状
    # 计算复杂度
    mla_flops, mla_params = get_model_complexity_info(wrapped_mla, input_res=(bsz, seqlen, args.dim), as_strings=True,print_per_layer_stat=True, verbose=True)
    print(f'MLA mla_flops:  {mla_flops}')
    print(f'MLA mla_params: {mla_params}')
    return mla_flops, mla_params

def cal_mlp(args):
    """
    计算 mlp
    """
    d = MLP(args.dim, args.inter_dim) #dim=7168,inter_dim=18432
    num_tokens = 1
    mlp_flops, mlp_params = get_model_complexity_info(d, (1,num_tokens,args.dim),as_strings=True,print_per_layer_stat=True)
    print(f"MLP mlp_flops:  {mlp_flops}")
    print(f"MLP mlp_params: {mlp_params}")
    return mlp_flops, mlp_params

def cal_moe(args):
    """
    计算 moe
    """
    # 计算Expert复杂度
    e = Expert(args.dim, args.moe_inter_dim) #dim=7168,moe_inter_dim=2048  
    moe_flops, moe_params = get_model_complexity_info(e, (1,num_tokens,args.dim),as_strings=True,print_per_layer_stat=True)
    print(f'MOE moe_flops:  {moe_flops}')
    print(f'MOE moe_params: {moe_params}') 
    return moe_flops, moe_params 

def main(
) -> None:
    with open('../configs/config_671B.json', 'r', encoding='utf-8') as f:
        with torch.device("cuda"):
            # 创建模型前确保默认类型
            torch.set_default_dtype(default_dtype)
            # print(f"Loading config from {f.read()}")
            args = ModelArgs(**json.load(f))

            cal_mla(args)
            cal_mlp(args)
            cal_moe(args)


if __name__ == "__main__":
    main()