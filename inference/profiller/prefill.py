import sys
from pathlib import Path
# 添加父目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

from model import ModelArgs
from gpu_perf import GPU_perf
from cal_params import cal_mla_flops, default_dtype
import torch
import json
from gpu_perf import gpu
import pandas as pd

def prefill_mla_elapse_time(args:ModelArgs, gpu:GPU_perf, seq_len, kv_cache_rate):
    """
    gemm_fp8_flops单位是GFLOPS
    gpu.get_fp8_flops单位是TFLOPS
    所有gemm_fp8_time单位是 ms,不是 s
    """
    _ , gemm_fp8_flops, attn_fp16_flops = cal_mla_flops(seq_len,seq_len,args, kv_cache_rate)
    gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops()
    print("GEMM_FP8 Elapsed time(ms): %.3f" % gemm_fp8_time)
    attn_fp16_time = attn_fp16_flops / gpu.get_fp16_flops()
    print("ATTN_FP16 Elapsed time(ms): %.3f" % attn_fp16_time)
    total_time = gemm_fp8_time + attn_fp16_time
    print("Total Elapsed time(ms):%.3f" % total_time)
    
    # 计算 MLA 的 All-Reduce 通信耗时：
    """
    这里计算可能有错误？？
    all_reduce 包含发送 + 接收，总通信量应该是 seq_len * args.dim * 2; 再算上 fp16,应该是 seq_len * args.dim * 2 * 2
    """
    all_reduce_comm_size = seq_len * args.dim * 2 /1024/1024
    ar_elapsed_time = all_reduce_comm_size / gpu.get_nvlink_bw() # 除以 NVLink 带宽（单位 MB/s）
    print("AR Elapsed time(ms):%.3f" % ar_elapsed_time)
    tp4_time = total_time/4 + ar_elapsed_time
    print("TP4 Elapsed time(ms):%.3f" % tp4_time)
    tp8_time = total_time/8 + ar_elapsed_time
    print("TP8 Elapsed time(ms):%.3f" % tp8_time)
    return total_time, tp4_time,tp8_time

def prefill_mla(args:ModelArgs, gpu_dict, seq_len, kv_cache_rate):
    """
    计算不同卡上的计算耗时
    pandas 是一个 python 数据分析包
    """
    df = pd.DataFrame(columns=['GPU','TP1','TP4','TP8'])
    for key in gpu_dict.keys():
        print('------------ %s --------------' % key)
        tp1,tp4,tp8 = prefill_mla_elapse_time(args,gpu_dict[key], seq_len, kv_cache_rate)
        df.loc[len(df)]=[key,tp1,tp4,tp8]
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))
    return 3 * seq_len * args.dim * args.inter_dim *2/1e9

def densmlp_flops(args:ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.inter_dim *2/1e9

def dense_mlp_elapse_time(args:ModelArgs,gpu:GPU_perf, seq_len):
    gemm_fp8_flops = densmlp_flops(args, seq_len)
    gemm_fp8_time = gemm_fp8_flops / gpu.get_fp8_flops()
    print("Elapsed time(ms): %.3f" % gemm_fp8_time)
    return gemm_fp8_time

def prefill_dense_mlp(args:ModelArgs, gpu_dict, seq_len):
    df = pd.DataFrame(columns=['GPU','DenseMLP耗时'])
    for key in gpu_dict.keys():
        print('------------ %s --------------' % key)
        t = dense_mlp_elapse_time(args,gpu_dict[key], seq_len)
        df.loc[len(df)]=[key,t]
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))


def moe_expert_flops(args:ModelArgs, seq_len):
    return 3 * seq_len * args.dim * args.moe_inter_dim *2/1e9

def moe_expert_elapse_time(args:ModelArgs,gpu:GPU_perf, seq_len, tp, dp):
    """
    TP=4时, DP=8, 那么相当于MLA同时产生了8组seq_len的token, 平均每卡Shared Expert计算的token数为 seq_len* dp_group / num_gpu
    这里实际上假设了 TP 对 MOE 参数计算部分也做了拆分，否则每个 dp 组内的 gpu 应当收到相同的seq_len
    在 sglang python/sglang/srt/models/deepseek_v2.py 中，确实 MOE 有 tp 拆分
    
    什么场景下会需要在 MOE 中做 tp 拆分呢？
    就是单个专家太大了，一张卡放不下来，才会需要引入 tp
    如果专家很小，单卡能放下，那每个专家就只在一张卡上，这时没必要再用 TP，直接 EP 就行
    """
    num_device = tp * dp # 所有节点有多少卡
    num_shared_token = dp * seq_len / num_device
    shared_flops = moe_expert_flops(args, num_shared_token)
    shared_time = shared_flops / gpu.get_fp8_flops()
    print("Shared Expert Elapsed time(ms): %.3f" % shared_time)

    num_routed_token = seq_len * dp * args.n_activated_experts / num_device
    routed_flops = moe_expert_flops(args, num_routed_token)
    routed_time = routed_flops / gpu.get_fp8_flops()
    print("Routed Expert Elapsed time(ms): %.3f" % routed_time)

    return shared_time, routed_time

def prefill_moe(args:ModelArgs, gpu_dict, seq_len, tp, dp ):
    df = pd.DataFrame(columns=['GPU','Shared Expert','Routed Expert'])
    for key in gpu_dict.keys():
        print('------------ %s --------------' % key)
        s, r = moe_expert_elapse_time(args,gpu_dict[key], seq_len,tp,dp)
        df.loc[len(df)]=[key,s,r]
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))


def prefill_alltoall_time(args:ModelArgs, gpu, seq_len, dispatch_node, tp):
    ##通信量估计
    gpu_per_node = 8
    dp = gpu_per_node/tp
    dispatch_size = (dispatch_node - 1) * dp * seq_len * args.dim /1024/1024
    combine_size = 2 * dispatch_size  #fp16  
    comm_bw = gpu.get_pcie_bw() * gpu_per_node
    dispatch_time = dispatch_size / comm_bw
    combine_time = combine_size / comm_bw
    return dispatch_time, combine_time


def prefill_alltoall(args:ModelArgs, gpu_dict, seq_len, dispatch_node, tp):  
    df = pd.DataFrame(columns=['GPU','Dispatch','Combine'])
    for key in gpu_dict.keys():
        print('------------ %s --------------' % key)
        dispatch_time, combine_time = prefill_alltoall_time(args, gpu_dict[key],seq_len, dispatch_node, tp)
        print("Dispatch Elapsed time(ms): %.3f" % dispatch_time)
        print("Combine Elapsed time(ms): %.3f" % combine_time)      
        df.loc[len(df)]=[key,dispatch_time,combine_time]
    print(df.set_index('GPU').to_markdown(floatfmt=".3f"))



def prefill_time(args:ModelArgs, gpu, seq_len, kv_cache_rate, tp , dp):
    dispatch_node = 4
    gpu_per_node = 8
    num_device  =  tp * dp
    dense_mla,tp4_mla,tp8_mla = prefill_mla_elapse_time(args, gpu,  seq_len, kv_cache_rate) 
    tp_mla = tp4_mla if tp == 4 else tp8_mla
    dense_mlp = dense_mlp_elapse_time(args, gpu, seq_len)
    shared, routed = moe_expert_elapse_time(args, gpu, seq_len, tp, dp)
    dispatch, combine = prefill_alltoall_time(args, gpu, seq_len, dispatch_node, tp)
    return dense_mla, dense_mlp, tp_mla, shared, routed, dispatch, combine
    
def prefill_time_sum(args:ModelArgs, gpu_dict, seq_len, kv_cache_rate, tp , dp):
    df = pd.DataFrame(columns=['MLA','DenseMLP','TP_MLA','Shared Expert','Routed Expert','Dispatch','Combine','GPU'])
    df2 = pd.DataFrame(columns=['Sum(Overlap)','Sum','GPU'])
    n_sparse_layers = args.n_layers - args.n_dense_layers
    df.loc[len(df)]= [ args.n_dense_layers, args.n_dense_layers,  #MLA+ DenseMLP
                       n_sparse_layers, n_sparse_layers, n_sparse_layers, #SparseLayer MLA + MoE
                       n_sparse_layers, n_sparse_layers, 'Layers'] #Dispatch & Combine Layers
    for key in gpu_dict.keys():
        t  = list(prefill_time(args, gpu_dict[key], seq_len, kv_cache_rate , tp , dp))
        t.append(key)
        df.loc[len(df)]= t
        sum_overlap = args.n_dense_layers * (t[0] + t[1]) + n_sparse_layers * ( t[2] + t[3] + t[4]) 
        sum_non_overlap = sum_overlap + n_sparse_layers * ( t[5] + t[6]) #alltoall
        df2.loc[len(df2)]= [ sum_overlap, sum_non_overlap, key]
    df = df.set_index('GPU').T
    df['Layers'] = df['Layers'].astype(int).astype(str)
    print(df.to_markdown(floatfmt=".3f"))  
    print('-----------SUM-------------')
    df2 = df2.set_index('GPU').T
    print(df2.to_markdown(floatfmt=".3f"))  
    
    return df,df2


if __name__ == "__main__":
    with open('../configs/config_671B.json', 'r', encoding='utf-8') as f:
        with torch.device("cuda"):
            # 创建模型前确保默认类型
            torch.set_default_dtype(default_dtype)
            # print(f"Loading config from {f.read()}")
            args = ModelArgs(**json.load(f))

            seq_len = 4383
            kv_cache_rate = 0.563

            # prefill_mla(args, gpu, seq_len,kv_cache_rate)
            # print('\n\n\n')
            # prefill_dense_mlp(args, gpu, seq_len)
            # print('\n\n\n')
            # prefill_moe(args, gpu, seq_len, tp=4,dp=8)
            # print('\n\n\n')
            # prefill_alltoall(args,gpu,seq_len,dispatch_node=4,tp=4)
            # print('\n\n\n')
            tp4_detail,tp4_sum = prefill_time_sum(args, gpu, seq_len, kv_cache_rate,tp=4 , dp=8)
            # tp4_detail,tp4_sum = prefill_time_sum(args, gpu, seq_len, kv_cache_rate,tp=8 , dp=4)