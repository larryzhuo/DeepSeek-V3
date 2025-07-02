import sys
from pathlib import Path
# 添加父目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from model import ModelArgs
from gpu_perf import GPU_perf
from cal_params import cal_mla_flops, default_dtype, mla_matabsob_flops
import torch
import json
from gpu_perf import gpu_decode,gpu_decode2
import pandas as pd
from prefill import dense_mlp_elapse_time, moe_expert_flops
from functools import reduce

class MoE_EP():
    def __init__(self,args:ModelArgs,ep_num, redundant_exp):
        self.ep_num = ep_num
        self.redundant_exp = redundant_exp
        self.dispatch_num = args.n_activated_experts
        self.n_routed_experts = args.n_routed_experts
        self.expert_num = (args.n_routed_experts + redundant_exp) / self.ep_num

    def expert_per_gpu(self):
        return self.expert_num
        
    def total_tokens(self,bs):
        return bs * self.ep_num

    def comm_tokens(self, bs):
        #平均每个token有self.expert_num / self.n_routed_experts概率本地处理 
        return bs * self.dispatch_num *(1- self.expert_num / self.n_routed_experts)
        
    def compute_tokens(self, bs):
        #总token数为bs * dispatch_num * ep_num, 平摊到每张卡/ep_num
        return bs * self.dispatch_num  




def _decoding_batchsize(args:ModelArgs, gpu:GPU_perf, seq_len,decode_len,tp, expert_num, absorb=True, kvcache_fp16=False):
    mem_util_rate = 0.9 #torch/activation等其它开销的折扣
    mla = 598.14 if absorb else 187.17 #MLA的参数(单位M)
    expert_mem = 44.05 #expert的参数(单位M)
    others_parameter = 2.91 #其它参数2.91GB
    kv_cache = (seq_len+decode_len) * (args.kv_lora_rank + args.qk_rope_head_dim) *args.n_layers *tp
    if kvcache_fp16 :
        kv_cache *=2
    mem = gpu.mem * mem_util_rate - others_parameter - mla * args.n_layers/tp/1024
    mem -= expert_mem *(args.n_layers - args.n_dense_layers) * expert_num /1024
    return mem * 1024 * 1024 * 1024 / kv_cache

def decode_batchsize(args:ModelArgs, gpu_dict, seq_len,decode_len, tp):
    df = pd.DataFrame(columns=['GPU','EP320','EP144','EP72','EP34'])
    for fp16_kvcache in range(0,2):
        for key in gpu_dict.keys():
            for absorb in range(0,2):
                item = key
                if bool(fp16_kvcache):
                    item +='_FP16'
                else:
                    item +='_FP8'
                if bool(absorb):
                    item +='_Absorb'                    
                value = [item]
                for exp_num in [2,3,5,9]:
                    bs = _decoding_batchsize(args, gpu_dict[key], seq_len,decode_len, tp,exp_num, bool(absorb),bool(fp16_kvcache))
                    value.append(bs)
                df.loc[len(df)]= value
    print(df.set_index('GPU').to_markdown(floatfmt=".0f"))  
    return df




bs_list =[32, 64, 128, 256]

def decode_mla_elapse_time(args:ModelArgs, gpu:GPU_perf, seq_len, bs, absorb=True):
    mla_flops_func = mla_matabsob_flops if absorb else cal_mla_flops
    #Decoding时计算为qlen=1, kv_cache_rate = 1
    _ , gemm_fp8_flops, attn_fp16_flops = mla_flops_func(1,seq_len,args, 1)
    
    gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops() * bs
    print("GEMM_FP8 Elapsed time(ms): %.3f" % gemm_fp8_time)
    attn_fp16_time = attn_fp16_flops / gpu.get_fp16_flops() *bs
    print("ATTN_FP16 Elapsed time(ms): %.3f" % attn_fp16_time) 
    total_time = gemm_fp8_time + attn_fp16_time
    print("Total Elapsed time(ms):%.3f" % total_time)
    all_reduce_comm_size = seq_len * args.dim * 2 /1024/1024  #fp16 take 2Bytes
    ar_elapsed_time = all_reduce_comm_size / gpu.get_nvlink_bw()
    print("AR Elapsed time(ms):%.3f" % ar_elapsed_time)
    tp4_time = total_time/4 + ar_elapsed_time
    print("TP4 Elapsed time(ms):%.3f" % tp4_time)
    tp8_time = total_time/8 + ar_elapsed_time
    print("TP8 Elapsed time(ms):%.3f" % tp8_time)
    return total_time, tp4_time, tp8_time

def decode_kvcache_load_time(args:ModelArgs, gpu:GPU_perf, seq_len, bs):
    kv_cache = seq_len * (args.kv_lora_rank + args.qk_rope_head_dim)  * bs 
    load_kv_time = kv_cache /1024/1024/1024 / gpu.get_mem_bw() *1000
    return load_kv_time     

def decode_mla(args:ModelArgs, gpu_dict, seq_len,absorb=True):
    df = pd.DataFrame(columns=['GPU','BatchSize','TP1','TP4','TP8','LoadKV_FP8','LoadKV_FP16'])
    for key in gpu_dict.keys():
        for bs in bs_list: 
             tp1, tp4,tp8 = decode_mla_elapse_time(args,gpu_dict[key], seq_len, bs,absorb)
             kv = decode_kvcache_load_time(args,gpu_dict[key], seq_len, bs)
             df.loc[len(df)]= [key, bs,tp1,tp4,tp8,kv, kv*2]
             df['BatchSize'] = df['BatchSize'].astype(int).astype(str)
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))  
    return df

def decode_dense_mlp(args:ModelArgs, gpu_dict):
    df = pd.DataFrame(columns=['GPU','BatchSize','DenseMLP'])
    for key in gpu_dict.keys():
        for bs in bs_list: 
            t = dense_mlp_elapse_time(args,gpu_dict[key], bs)
            df.loc[len(df)]=[key,bs,t]
    df['BatchSize'] = df['BatchSize'].astype(int).astype(str)        
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return df
    

def _moe_expert_time(args:ModelArgs,gpu:GPU_perf,bs):
    group_gemm_discount_rate = 0.7
    shared_flops = moe_expert_flops(args, bs)
    shared_time = shared_flops / gpu.get_fp8_flops() / group_gemm_discount_rate

    num_routed_token = bs * args.n_activated_experts
    routed_flops = moe_expert_flops(args, num_routed_token)
    routed_time = routed_flops / gpu.get_fp8_flops() / group_gemm_discount_rate
    return shared_time, routed_time

def moe_expert_time(args:ModelArgs,gpu_dict):
    df = pd.DataFrame(columns=['GPU','BatchSize','SharedExpert','RoutedExpert'])
    for gpu_key in gpu_dict.keys():
        for bs in bs_list: 
            s, r = _moe_expert_time(args,gpu_dict[gpu_key], bs)
            df.loc[len(df)]=[gpu_key,str(bs),s,r]
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))        
    return df


def _moe_a2a(args:ModelArgs,gpu:GPU_perf,bs):
    dispatch_size = bs * args.dim * args.n_activated_experts /1024/1024 
    combine_size = dispatch_size * 2 #FP16
    dispatch_t = dispatch_size / gpu.get_pcie_bw()
    combine_t = combine_size / gpu.get_pcie_bw()
    return dispatch_t, combine_t


def decode_a2a(args:ModelArgs, gpu_dict):  
    df = pd.DataFrame(columns=['GPU','BatchSize','Dispatch','Combine'])
    for key in gpu_dict.keys():
        for bs in [64, 128, 256]: 
            dispatch_time, combine_time = _moe_a2a(args, gpu_dict[key],bs)
            df.loc[len(df)]=[key,str(bs),dispatch_time,combine_time]
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))


def _decoding_time(args:ModelArgs, gpu:GPU_perf,seq_len):
    mla = decode_mla(args,gpu,seq_len)
    dense_mlp = decode_dense_mlp(args,gpu)
    moe = moe_expert_time(args,gpu)
    a2a = decode_a2a(args,gpu)
    dfs = [ mla, dense_mlp, moe, a2a]
    df = reduce(lambda left, right: pd.merge(left,right, on=['GPU','BatchSize'], how='left'), dfs)
    print(df.set_index('GPU').T.to_markdown(floatfmt=".3f"))
    return df




if __name__ == "__main__":
    with open('../configs/config_671B.json', 'r', encoding='utf-8') as f:
        with torch.device("cuda"):
            # 创建模型前确保默认类型
            torch.set_default_dtype(default_dtype)
            # print(f"Loading config from {f.read()}")
            args = ModelArgs(**json.load(f))

            seq_len = 4383

            ep_dict = { 'EP34': MoE_EP(args, 34,16),
            'EP72' :MoE_EP(args, 72,32),
            'EP144' :MoE_EP(args, 144,32),
            'EP320' :MoE_EP(args, 320,64)}

            decode_len = 1210
            df = decode_batchsize(args,gpu_decode, seq_len,decode_len, tp=1)
            print('\n\n\n')
            decode_mla(args,gpu_decode,seq_len)
            print('\n\n\n')
            decode_dense_mlp(args,gpu_decode)
            print('\n\n\n')
            moe_expert_time(args,gpu_decode)
            print('\n\n\n')
            decode_a2a(args,gpu_decode)
            print('\n\n\n')
            # dfs = _decoding_time(args,gpu_decode2,seq_len)
            print('\n\n\n')

