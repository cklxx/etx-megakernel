# DeepSeek 多 batch 归因, 2026-10-18

数据：`examples/gpu-live/results/2026-10-16-multibatch/trace_report.json`（一台 MI300X
VF，gfx942，ROCm 7.14；B1 arm `alld4r0l1kvsmw`，B2/B4 arm `x2`/`x4`；bf16，context
1024，32 token，每 batch 3 轮；这里用 round 1，另两轮的 step 差异 ≤0.221%）。报告
`status` 为 COMPLETE，覆盖 9 次运行（B1/B2/B4 × 3 轮）。B8 没有 trace，见文末。

## 结果

| Batch | step (us) | tokens/s | 关键路径 body (us) | 依赖间隙 (us) | 排队间隙 (us) | 任务数 |
|---|---|---|---|---|---|---|
| 1 | 3254 | 307 | 2716 (83%) | 378.7 | 0.0 | 53022 |
| 2 | 4576 | 437 | 3735 (82%) | 542.2 | 69.7 | 109892 |
| 4 | 9027 | 443 | 7360 (82%) | 955.1 | 381.6 | 219784 |

三个关键路径列相加等于 trace 的 span（B1/B2/B4 = 3094/4347/8697 us），不是 step；
step 比 span 多 160/229/330 us，是 trace 之外的启动与收尾开销。

吞吐在 B2 就基本到顶：B1→B2 +42.2%（307→437），B2→B4 只有 +1.4%（437→443）。

## 归因：为什么 B4 不再涨

B2→B4 的 step 增量是 +4450.5 us；其中 trace 覆盖的 span 只增加 +4349.8 us。
以下百分比都按 step 增量计算：

- **body +3624.9 us（81%）**。任务数严格 2 倍（109892→219784），body 也几乎严格 2 倍
  （3735.3→7360.2，1.97×）。这说明每个请求复制整张任务图，trace 中**看不到跨请求的
  权重摊销**；body 随 batch 线性长，是设计结果，不是异常。
- **依赖间隙 +412.9 us（9%）**。`router` 78.0→183.5、`oproj` 81.7→181.4、
  `qabs` 80.5→172.1 是主要来源。`qabs` 的依赖等待只在 B≥2 进入关键路径：B1 有
  864 个 `qabs` 任务但 CP 贡献为 0；B4 总计 172.1 us，B2→B4 新增 91.6 us。
- **排队间隙 +312.0 us（7%）**。几乎全部来自 `qkv`（0→225.2 us）和 `attn`
  （69.7→148.1 us），两者合计 +303.6 us；其余 +8.4 us 来自 `lm_head`（+4.6）和
  `embed`（+3.7）。4 个请求的两族任务争同一批 CU，最后的任务要等前面的算完。
- **trace 之外 +100.8 us（2%）**。step 增量 +4450.5 us 与 span 增量 +4349.8 us
  的差，是 trace 未覆盖的启动/收尾开销。

## 机制结论

1. **吞吐上限来自“复制整图”而不是调度**：body 占 step 的 82%，并随 batch 近似
   线性增长；B2→B4 的调度间隙（依赖 + 排队）只占 step 增量的 16%，其余是复制的
   body 工作。所以 B4 相对 B2 只多出 1.4%，不是调度浪费，而是没有可摊销的共享工作。
2. **B≥2 新增的串行点是 `qabs` 的依赖等待**：B4 在关键路径上总计 172.1 us
   （B2 为 80.5 us），B1 虽然也有 864 个 `qabs` 任务，但不在关键路径上（0 us）。
   这是 B2 之后才付出的固定成本；172.1 us 是整个 B4 step 的总和，不是每请求值。
3. **真正的排队损耗在 `qkv` 与 `attn`**：这两族在 B4 的排队间隙合计 373.3 us
   （qkv 225.2 + attn 148.1），占 step 的 4.1%；要提升 B4 吞吐，先看这两族的
   并发度，而不是 body 内部。

## 还缺什么

机制清单与记录的覆盖情况见 `docs/deepseek/DS-BATCH-MECHANISM-SCAFFOLD-2026-10-18.md`（标 [M]/[U]）。
记录中没有 HBM/L2 流量、硬件 occupancy 计数、launch/dispatch 计数和任务级有序依赖链。

- **B8 没有 trace。** 已搜索本仓库的所有本地/远程分支和 `~/.hermes/cache/scratch`：
  唯一的 `trace_report.json` 就是本文这份，只含 B1/B2/B4（9 次运行）；2026-10-14 的
  B8 产物只有 gate/build/plan 收据。全部 8 份 `bgate_x8*.json` 都是 NOT COMPLETE，
  同一个错误 "request 5: 3 layers within the gate (need 27)"，例如
  `examples/dsv2lite/results/2026-10-14-batch-20261014T202530Z/20261014T202530Z-898d2b/bgate_x8.json`。
  scratch 中名为 `b8` 的文件属于 Qwen/GLM 的 CPU 验收，不是 DeepSeek trace。
  因此没有可归因的 B8 trace，上面的结论不外推到 B8。
- 没有每个 task 的 wave 时间戳，所以“谁在等谁”只能从间隙量级推断，不能逐任务归责。
- 跨请求共享权重（一次加载、两次使用）没有测；按 body 线性推测，它能改变的是 81% 那一项，
  但需要新计划和新测量才能算收益。

## 复现

```bash
python3 tools/ds_batch_attribution.py examples/gpu-live/results/2026-10-16-multibatch/trace_report.json
.venv/bin/python -m pytest -q tests/test_ds_batch_attribution_tool.py
```

输出与上面的表一致（每 batch 的 step、tok/s、body/dep/queue 拆分，以及相邻 batch
的分族增量）。测试用一份合成 trace 固定同一组算式，并回读真实 receipt 校验本文的
B1/B2/B4 数值。
