# Gains table

S1: `results_nvref_2026-10-09`, AMD Instinct MI300X VF, gfx942, ROCm 7.2.4, kernel 6.8.0-124, 8 cores.
S2: `results_nvref2_2026-10-09/session-7de5pood`, AMD Instinct MI300X VF, gfx942, ROCm 7.14.0, kernel 6.8.0-138, 20 cores.
Both: Qwen3-30B-A3B bf16, batch 1, prompt 1024, 32 greedy tokens, 32/32 equal to HF greedy; same vLLM image `sha256:bb44b39a…`; same plan.
Rule: only paired differences inside one session compare.

| 比较 | 数值 | 条件 | 状态 |
|---|---|---|---|
| 融合 对 同一组 kernel 逐个经 HIP graph 启动 | S1：设备 +0.225 ms 4.87%，主机多耗 0.094 ms，墙钟 +0.130 ms 2.79%。S2：设备 +0.262 ms 5.71%，主机多耗 0.100 ms，墙钟 +0.162 ms 3.51% | 每次 5 对交错，GPU 0，同一会话内同一计时构建，4 个 tile flag；主机项是 RESULT 中位数代理，不能直接相减还原墙钟；规则：每对为正，中位 >2% 且大于两倍离散，两次都满足 | 已复现（计时规则） |
| ETX 对 vLLM 默认，墙钟，预注册判定 | S2：LEAD，配对差 -0.449 ms，约 9.1%，离散 0.036 ms。S1：INSUFFICIENT，配对差 -0.283 ms，离散 1.013 ms | vLLM 0.27.1 默认；每次 5 轮交错，S2 5/5 轮 32/32；规则：5 轮全负，中位 ≤ -0.050 ms，中位绝对值大于离散 | 实测 |
| ETX 对 vLLM 默认，墙钟，事后估计 | 四轮差值均值 -0.295 ms，约 6.1% [E] | 仅 S1，n = 4；排除规则：第 2 轮 one_ms 48.57 ms，落在健康区间 17.18 到 18.12 ms 之外；规则在看到数据后才定 | 估计 |
| vLLM 估计器 §12 amendment | S1：每轮 1 个样本，没有轮内离散；轮间离散 1.009 ms，9 个健康轮 0.049 ms。S2：每轮取 5 个样本的中位，轮内 0.008 到 0.021 ms，轮间 0.032 ms | §12 预测 MISSED：轮间 0.032 ms，超过 ≤ 0.020 ms [E] 的预测 bar；配对差离散 0.036 ms，达到 0.050 ms 规则 | 实测 |
| 模型误差，profile 2026-10-08 | 对 S1：融合设备 -0.335 ms，-7.6%；融合增益误差 -0.037 ms。对 S2：-0.272 ms，-6.3% [E]；增益误差 -0.074 ms [E] | CPU 模拟，同一 plan，预测 4.058 ms；门槛：±10% 以内，增益误差 < 0.093 ms；S1 两项通过 | 估计 |
| 两次都没定下的事 | 5 个导入实例做过字节校验，共 97 个；模型残差 -0.335 ms，原因未知 | 两次都只在 chk1 构建上校验，计时构建拒绝校验；vLLM 只测了默认和 AITER；S2 的 trace1、floors_trace、evx 被跳过；LEAD 只出现一次；vLLM 计时含采样和调度，ETX 不含，这部分未测 | 未定 |

两次都是 MI300X VF，但 ROCm 版本不同，所以绝对毫秒不能跨会话比较，只比同一会话内的配对差值。


**LEAD 仅一次、仅 VF。** `S2/report.log` 的 NOT-COMPLETE 是读错旧 status 的结果。

**LEAD 来源：** 对 S2 目录重新运行 `nvref_report`，输出在 `results_nvref2_2026-10-09/report.log`。

**In review：** logdir 修复在另一分支审查中。

**3.51%** 用 31 步墙钟均值；`pairs.py` 用 RESULT 中位数得到 **3.50%**，两者均通过。

**0.032 ms** 来自日志中已舍入的轮中位数；完整 JSON 是 **0.032632 ms**，同样 MISSED。

**4.920、4.921、4.911、4.899、4.889 ms** 是 §12 的五个报告中位数。

**LEAD resolved：** 配对差离散 0.036 ms 在 0.050 ms 内；这不等于预测达到。

| 表中行 | 可重算的来源 |
|---|---|
| 融合与主机项 | S1 `report.log`、`fusion_pairs.txt`；S2 `etx_r1.log` … `etx_r5.log`、`etxu_r1.log` … `etxu_r5.log` 的步记录及 RESULT；[重算命令](NVREF2-2026-10-09.md#measurements-m) |
| 预注册系统判定 | S1 `report.log`；S2 `vllm_r1.log` … `vllm_r5.log` 与 `etx_r1.log` … `etx_r5.log`；重新渲染的 `results_nvref2_2026-10-09/report.log` |
| S1 事后估计 | S1 `vllm_r1.json` … `vllm_r5.json`、`report.log`；[原记录](NVREF-2026-10-09.md#2-paired-verdict-and-fusion) |
| §12 amendment | S1 `vllm_r1.json` … `vllm_r5.json`、`vs_r1.json` … `vs_r6.json`；`results_nvref2_2026-10-09/vllm_r1.json` … `vllm_r5.json`；S2 五个 `vllm_r*.log` |
| 模型误差 [E] | [固定 profile 预测](MODEL-vllm-kernels.md#21-fixed-2026-10-08-tile-profile-cpu-validation-2026-10-09)：4.058 ms、增益 0.188 ms；S1 `report.log`、`fusion_pairs.txt`；S2 五对 RESULT；S2 算式 4.058−4.330、0.188−0.262 |
| 未定事项 | S1 `check_imp1.txt`、`vlplan_all.txt`；S2 `check_imp1.log`、`check_imp.log`、`build_chk1.log`、`status.json`；`results_nvref2_2026-10-09/vlplan_all.txt`；两次报告 residual 行 |
