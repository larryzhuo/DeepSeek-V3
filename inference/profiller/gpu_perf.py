class GPU_perf():
    """
    - 注: 算力单位为 TFLOPS,带宽单位为GB/s
    TFLOPS（Tera Floating-Point Operations Per Second）是衡量计算设备（如 GPU、CPU）浮点运算能力的单位，表示 每秒万亿次浮点运算
    1 TFLOPS = 1e12 FLOPS
    1 GFLOPS = 1e9 FLOPS
    """
    def __init__(self,sm,comm_sm, fp16_flops,fp8_flops,mem,mem_bw, nvlink_bw,pcie_bw, discount_rate):
        self.sm = sm
        self.comm_sm = comm_sm #用于通信的SM数量
        self.fp16_flops = fp16_flops
        self.fp8_flops = fp8_flops
        self.mem = mem
        self.mem_bw = mem_bw
        self.nvlink_bw = nvlink_bw
        self.pcie_bw = pcie_bw
        self.discount_rate = discount_rate #整体性能按峰值性能折扣
        #TODO: 可以分离网络性能折扣和算力性能折扣

    def get_fp16_flops(self):
        """
        这里返回的单位是 TFLOPS
        """
        return self.fp16_flops * self.discount_rate  * ( self.sm  - self.comm_sm) / self.sm

    def get_fp8_flops(self):
        """
        这里返回的单位是 TFLOPS
        """
        return self.fp8_flops *  self.discount_rate * ( self.sm  - self.comm_sm) / self.sm

    def get_mem_bw(self):
        return self.mem_bw *  self.discount_rate

    def get_nvlink_bw(self):
        return self.nvlink_bw *  self.discount_rate

    def get_pcie_bw(self):
        return self.pcie_bw *  self.discount_rate

h800 = GPU_perf( sm = 132 ,comm_sm = 24, 
                 fp16_flops = 791.6, fp8_flops = 1583.2, 
                 mem = 80,mem_bw = 3350,
                 nvlink_bw = 200,pcie_bw = 50,
                 discount_rate = 0.85)

h20 = GPU_perf( sm = 78 ,comm_sm = 10, 
                 fp16_flops = 118.4, fp8_flops = 236.8, 
                 mem = 96,mem_bw = 3350,
                 nvlink_bw = 400,pcie_bw = 50,
                 discount_rate = 0.85)
h20_3e = GPU_perf( sm = 78 ,comm_sm = 0, 
                 fp16_flops = 118.4, fp8_flops = 236.8, 
                 mem = 141,mem_bw = 4800,
                 nvlink_bw = 400,pcie_bw = 50,
                 discount_rate = 0.85)

gpu = dict({'H800': h800, 'H20': h20})
gpu_decode = dict({'H800': h800, 'H20': h20,'H20_3e': h20_3e})
gpu_decode2 = dict({'H800': h800, 'H20': h20}) 