# 动态 Megakernel 编译器 · 完整架构设计

> 目标系统代号 **ETX**（Event Tensor eXtended）。
> 血缘：Event Tensor / ETC 的抽象 + 我们自己补的机器模型与代价模型。
> 版本：v0.1 设计稿，写于 2026-09-22。状态：待评审，未实现。
> **v0.2（2026-09-22 下午）已改为单页 `public/index.html`**，新增：2026 生态十条设计轴、Tile ABI 与 link / host-DSL 两种接入模式、按 scope 的 fence / 轮询 lowering 表（来自 LLVM gfx942 memory model）、公开测得的同步代价种子值、LDS 分页、epoch 事件与跨步控制环、调度策略交互模拟。本文件保留为 v0.1 原稿。

---

## 0. 一分钟版

一句话：把「多算子融合成一个持久化内核」这件事，做成一个 DSL 无关、硬件无关的编译器，
让同一份程序在 8 个 XCD 的 MI300X、2 个 GCD 的 MI250X、双 die 的 B200 上都生成该机器上合理的调度。

三条主线，永不混在一起：

| 线 | 内容 | 放在哪 | 错了会怎样 |
|---|---|---|---|
| 依赖语义 | 谁等谁、等几个、动态触发范围 | L2 IR | 结果错 / 死锁 |
| 机器事实 | 这台机器有什么能力、每次同步多少钱 | L3 机器模型 | 能力错=静默错，代价错=性能差 |
| 决策 | 事件张量怎么分片、用哪种调度 | L4 pass | 只是慢，不改语义 |

设计上的三个硬决定：

1. **硬件差异只能以数据形式存在**，代码路径里不许出现 `if (arch == gfx942)`。
2. **能力（capability）与代价（cost）分开建模**：能力决定能不能做，代价决定值不值得做。
3. **调度策略由代价模型选，不由用户猜**。种子证据：ETC 的动态调度在 TP=4 上只有 0.83x，比不融合还慢。

---

## 1. 设计目标与非目标

### 目标（按优先级）

1. **DSL 无关**：任何 tile 级前端（Triton、TileLang、CuteDSL、手写 tile IR）都能接。
2. **跨硬件可移植**：同一份 IR，在 CDNA2/CDNA3/CDNA4 与 NVIDIA Blackwell 上都能出正确、合理的代码。
3. **动态负载一等支持**：shape 动态 + 数据依赖动态，编译期不需要知道具体值。
4. **调度决策自动化**：编译器按代价表选静态/动态/混合，而不是让用户按直觉配置。
5. **可验证**：每个融合内核都有等价的未融合参考实现，能逐元素比对。

### 非目标（明确不做）

- 不做训练框架，只做推理侧的算子融合。
- 不自研 tile 级代码生成的全部技术，允许内联已调好的手写 tile（CUTLASS/CK 风格）。
- 第一版不做跨节点（多机）。IR 里不写死「单机」，但调度器先只考虑单机内拓扑。
- 不追求「所有算子都能融合」，支持融合失败的优雅退化（部分子图融合）。
- 不处理任意控制流。依赖必须能表达成仿射映射或范围映射。

---

## 2. 全景分层

```
┌──────────────────────────────────────────────────────────────┐
│ L1 前端适配器   Triton / TileLang / CuteDSL / 手写 tile IR    │
│    只回答三个问题：tile 坐标空间、依赖映射、资源需求            │
└───────────────────────────┬──────────────────────────────────┘
                            │ TileOp + EdgeMap（硬件无关）
┌───────────────────────────▼──────────────────────────────────┐
│ L2 Event-Tensor IR                                            │
│    task 网格、ETensor(shape, wait_count, scope)、符号维度      │
│    数据依赖更新 / 数据依赖触发；校验 pass（无环、计数守恒）      │
└───────────────────────────┬──────────────────────────────────┘
                            │ 带 ETensor 的 tile 依赖图
┌───────────────────────────▼──────────────────────────────────┐
│ L3 机器模型                                                   │
│    capability（能力位，布尔/枚举）→ 决定正确性                │
│    cost（代价表，数值）→ 决定性能                             │
└───────────────────────────┬──────────────────────────────────┘
                            │ 事件分片方案 / 调度模式 / 队列布局
┌───────────────────────────▼──────────────────────────────────┐
│ L4 放置与调度 pass                                            │
│    tile 划分 → 事件亲和性分片 → 调度模式选择 → 队列分层 → 预取 │
└───────────────────────────┬──────────────────────────────────┘
                            │ 持久化内核 + 描述符表 + 启动参数
┌───────────────────────────▼──────────────────────────────────┐
│ L5 后端 codegen + 极小运行时                                  │
│    notify/wait lowering 到各家原子指令；运行时只有两块状态      │
└──────────────────────────────────────────────────────────────┘
```

### 每层的产出物与验收标准

| 层 | 产出物 | 验收标准 |
|---|---|---|
| L1 | TileOp 描述（纯数据） | 换前端不改 L2 以下任何一行 |
| L2 | 带 ETensor 的 tile 依赖图 | 校验 pass 全绿；符号维度可实例化 |
| L3 | `<arch>.yaml` 能力表 + 代价表 | 新硬件只填表就能接后端 |
| L4 | 分片方案 + 调度模式 + 队列布局 | 决策可解释：打印「为什么选 static」 |
| L5 | 持久化内核源码 + 启动参数 | 逐元素对齐参考实现；无死锁 |

---

## 3. L1 前端契约：DSL 无关性怎么落地

### 3.1 最小契约

任何前端只要能回答下面三个问题，就能接入 ETX。
其余（内存布局、tile 内指令、warp 特化）都留在前端内部。

**问题一：tile 坐标空间是什么？**

```python
Grid = (dim_0_expr, dim_1_expr, ...)   # 允许符号表达式，如 (B * H, K // 128)
```

**问题二：每个 tile 的输入输出依赖是什么？**

```python
tile_coord -> set of (tensor_slice, producer_tile_coord)
```

ETX 不要求前端自己算依赖，允许两种给法：
- 显式：直接给出生产者坐标（手写 IR 用）
- 声明：给 reads/writes 的切片表达式，ETX 反推（Triton 类前端用）

**问题三：资源需求是多少？**

```python
Resource(threads, lds_bytes, vgpr, agpr, tensor_cores, smem_for_cluster?)
```

这三个问题的答案构成 `TileOp`，不含任何硬件细节。
`lds_bytes` 是抽象需求，具体怎么映射到 64KB 还是 160KB 是 L4 的事。

### 3.2 为什么这个契约足够

因为 megakernel 编译器只做三件事：切 tile、连依赖、排调度。
三件事各需要一份输入：坐标空间、依赖、资源。别的东西它不需要知道。

反过来说，如果某前端无法回答这三个问题（比如 tile 之间的依赖只能在运行时才知道且无法用仿射表达），
那它就不适合接入——这时应该退化到逐 kernel 执行，而不是硬塞进 ETX。

### 3.3 前端的两种接入深度

| 深度 | 前端提供 | 适用 |
|---|---|---|
| 浅接入 | 只有 TileOp，tile 内实现由 ETX 交给该前端的 codegen 完成 | Triton / TileLang |
| 深接入 | TileOp + 手写 tile 实现（内联汇编级） | CuteDSL / 手写 HIP |

浅接入的前端要保持「自己的 kernel 也能单独跑」，否则迁移过程中无法对比。
深接入的内核允许绕过 ETX 的 tile 生成，但必须仍提供资源需求，否则调度器无法决定 tile 尺寸。

---

## 4. L2 Event-Tensor IR 规范

### 4.1 核心类型

```
TensorIR  = (shape[符号或常量], dtype, layout_hint)
ETensorIR = (shape, wait_count, scope, domain_id, init_kind)
```

- `wait_count`：这个 event 需要被 notify 多少次才归零。可符号（如「路由到 expert i 的 token 数」）。
- `scope`：事件的可见性范围。`workgroup` < `cluster` < `agent` < `system`。
- `domain_id`：这个事件服务哪些 task 集合。分片决策由 L4 填。
- `init_kind`：静态常量 或 运行时计算（数据依赖场景）。

事件本身不会降低成新数据结构：**降成一张整型张量**，复用已有的张量机制与内存规划。

### 4.2 边映射表达式

依赖写成「task 坐标 → event 坐标」的映射。支持三种形式：

| 形式 | 例子 | 用途 |
|---|---|---|
| 仿射映射（einsum 子集） | `"ij->i"` | 规约、拆分、静态依赖 |
| 索引算术 | `"i->i*2"`、`"i->i/32"` | tile 尺寸不匹配的算子（如 RS tile 是 MM tile 的两倍） |
| 运行时范围 | `"i->range(indptr[i], indptr[i+1])"` | MoE：一个 producer 触发变数量的消费者 |
| 运行时映射 | `"i->topk[i,:]"` | MoE：producer 通知哪些 event 由运行时数据决定 |

**约束（这是编译器能做分析的前提）**：

- 映射必须是仿射的，或 `range(仿射, 仿射)` 形式。
- 禁止任意控制流进入边映射。
- 运行时值（`topk`、`indptr`）必须来自上游 tile 的写出，且带 release 语义。

### 4.3 语义

```
notify(ev)  ≡  atomic_dec(ev.count);  当结果 == 0 时，ev 被触发
wait(ev)    ≡  spin_until(ev.count == 0)
```

内存序：

- producer 的写必须在 `notify` 之前可见（release）
- consumer 的读必须在 `wait` 返回之后（acquire）

这两条是 IR 语义的一部分，不是后端实现细节。
后端如果发现目标硬件无法提供该 scope 的原子（见 5.3），必须**编译报错**，
而不是悄悄降级——因为降级会导致结果错，而且往往只在跨设备时才错。

### 4.4 动态性的两种表达

**shape 动态**：ETensor 与 Tensor 的维度可以是符号变量。
编译产物是模板，运行时用具体值实例化坐标空间。

**数据依赖动态**：靠 4.2 的 `range` 与 `topk` 映射。
关键在于：`wait_count` 也从静态变成运行时计算值。
初始化发生在算 `topk` 的那个 tile 内，与 `topk` 一起算出。

### 4.5 必须有的校验 pass

| 校验 | 检查什么 | 不做的话 |
|---|---|---|
| 无环 | 依赖图不能有环 | 死锁，且现场很难 debug |
| 计数守恒 | `wait_count` == 实际扇入次数（含运行时范围的上界） | 早触发或永不触发 |
| 范围边界先行 | `indptr`/`topk` 类张量必须先被 release 写出，再被 acquire 读 | 读到旧值，MoE 结果错但看起来正常 |
| tile 覆盖唯一 | 每个输出元素恰好被一个 tile 写 | 静默数据竞争 |
| 域合法性 | `scope=system` 的事件，其内存类型必须支持系统域原子 | 见 5.3，会被硬件静默降级 |

---

## 5. L3 机器模型

### 5.1 为什么必须有这一层

ETC 的抽象在 IR 层是硬件无关的，但它的实现与全部评测都只在 B200 上。
一旦拿到 MI300X，至少有四处会崩：

- 没有 multimem 指令（GEMM+Reduce-Scatter 的融合路径直接失效）
- L2 是 8 个各 4MB 的私有段，跨 XCD 只有 256MB MALL 是共享点
- cluster/DSMEM 支持按目标而定，不支持时 LLVM 把作用域退化成 agent scope
- 系统域原子对内存类型有要求，不满足会被静默降级

### 5.2 能力表（capability）：决定正确性

`capability` 是布尔与枚举，编译期必需知道，错了就是错的代码。

```yaml
# arch/gfx942.yaml（节选，字段全部来自公开文档，见文末来源）
name: gfx942
family: CDNA3
wave_size: 64
exec_domains:
  root: gpu
  children:
    - name: xcd
      count: 8
      cus_per: 38
memory:
  l2_per_domain_mb: 4          # 每个 XCD 私有 4MB
  l2_total_mb: 32
  shared_cache: { kind: infinity_cache, mb: 256 }
  lds_kb_per_cu: 64
  atomic_memory_type_required:   # 关键：不同 scope 对内存类型的要求
    agent:  any
    system: fine_grained         # coarse-grained 会被静默降级到 agent scope
capabilities:
  cluster_launch: unknown        # 按目标而定，必须实测
  distributed_shared_mem: false
  multicast_reduce: false        # 无 multimem 对应物
  atomic_rmw_forwarding: infinity_fabric   # 所有原子都转出到 IF
  fp_atomic_over_fabric: true    # MI300/MI350 支持；MI200 不支持
```

各代差异（会在仓库里各有一份 yaml）：

| 字段 | MI250X (gfx90a) | MI300X (gfx942) | MI350X/355X (gfx950) | B200 |
|---|---|---|---|---|
| exec domain | 2 GCD × 110 CU | 8 XCD × 38 CU = 304 | 8 XCD × 32 CU = 256 | 2 die，软件看成 1 颗 |
| L2 归属 | 8 MB / GCD | 4 MB / XCD（共 32 MB） | 4 MB / XCD（共 32 MB） | die 内统一 |
| 共享缓存 | 无 MALL | 256 MB Infinity Cache | 256 MB Infinity Cache | 统一 L2 |
| LDS / CU | 64 KB | 64 KB | 160 KB | ~228 KB |
| wave | 64 | 64 | 64 | 32 |
| cluster / DSMEM | 不支持 | 按目标而定 | 按目标而定 | 支持 |
| multicast / 归约写 | 无 | 无 | 无 | multimem ld_reduce |
| 浮点原子过互联 | **不支持**（过 IF 会 NOP） | 支持 | 支持 | 支持 |
| 带宽 | 3.2 TB/s | 5.3 TB/s | 8.0 TB/s | 8 TB/s 级 |

**注意最后两行**：MI200 的浮点原子不能过 Infinity Fabric，且若目标存储是 host memory 而 PCIe 不支持该原子，
硬件会退化成 load-op-store，极端情况下直接 NOP。这不是性能问题，是**静默错误**。
所以「这个原子的语义在目标内存上能否成立」必须是编译期可查的能力位。

### 5.3 系统域原子的三个陷阱（必须在文档里写死）

1. **coarse-grained 内存上的系统域原子会被降级为 agent scope。**
   事件张量若用于跨卡同步，分配时必须用 fine-grained。
2. **PCIe 不支持某原子时，GPU 侧变成 load-op-store。** 所有提交到该地址的 wave 会一起 stall，
   CPU 侧看到的是非原子序列。
3. **不支持的浮点原子在 uncached 内存上是 NOP**，结果错但不报错。

对应到我们的设计：**分配器（allocator）是 L4 的一部分，不是运行时细节。**
事件张量按 `scope` 分配内存类型，写错就是静默错，所以要有断言。

### 5.4 代价表（cost）：决定性能

三级同步代价，第一版按下面的模型，系数靠标定填：

```
T_sync(e) =
    T_local       若 producer 与 consumer 在同一 exec domain
    T_cross_domain 若跨 XCD/GCD（经 MALL / Infinity Fabric）
    T_cross_device 若跨卡（P2P ring / NVLink）
```

标定清单（这是阶段 0 的活）：

| 测什么 | 怎么测 | 数字喂给谁 |
|---|---|---|
| 同域原子 + 自旋延迟 | 单 XCD 内两个 CTA 来回 | `T_local` 系数 |
| 跨 XCD 原子 + 自旋 | 相邻 XCD / 最远 XCD 两档 | `T_cross_domain` |
| 经 MALL 与不经 MALL | 事件张量放 L2 常驻区 vs 普通 VRAM | MALL 是否值得做汇聚点 |
| 跨卡 P2P | 双卡 atomics + fence | `T_cross_device` |
| 排队列争用 | N 个 CTA 抢同一个队列，扫 N | 队列分层阈值 |

公开数字可作为量级参照：AMD 文档实测 MI300X/MI350X 上做 XCD-aware 的 program ID 重映射 + swizzle，
L2 miss 从约 5M 降到约 3.1M，多出约 67 TFLOPS。说明**纯放置决策就能带来 10% 量级的波动**。

### 5.5 调度模式的选择公式

设：

- `S_balance` = 动态调度带来的负载均衡收益（预计能省下的 straggler 时间）
- `C_cross` = 跨域 push 次数 × `T_cross_domain`（或跨卡时 `T_cross_device`）
- `C_queue` = 队列争用代价（由标定的争用曲线给出）

规则：

```
选 dynamic  当  S_balance > C_cross + C_queue
否则选 static（或 hybrid）
```

经验阈值的起点来自论文的两个数据点：

- MoE（同域、路由不规则、1024 token）：dynamic 1.08 vs static 1.04，选 dynamic
- 稠密 TP=4（跨卡）：dynamic 0.83 vs static 1.09，选 static

即：**跨域边一多，动态调度的收益就被 push 代价吃掉了。**

### 5.6 混合调度（hybrid）

这是论文没做的，也是我们最可能拿到优势的地方。

做法：把依赖图按 exec domain 切成若干 super-task。

- **域内**：用动态队列（局部队列，不跨域 push）。应对 MoE 式的域内不规则。
- **域间**：用静态长队列 + 事件计数。把跨域通信次数压到最小。

实现上，hybrid 不是新机制，而是 L4 对同一套原语的不同组装：
域内队列用 `push/pop`，域间边用 `notify/wait` + 预排队列。

---

## 6. L4 放置与调度 pass

五个 pass，按顺序：

### Pass 1 · tile 划分与资源配置

输入：TileOp + capability（LDS、寄存器、wave）。
输出：每个算子的 tile 尺寸与 tile 数。

约束：tile 的资源需求必须能装进目标 CU（注意 64KB vs 160KB 的 LDS 差异会改变可行 tile 尺寸）。

### Pass 2 · 事件亲和性分片

这是 chiplet 感知的核心 pass。

问题形式化：把 task/event 图切成 `k` 块（`k` = exec domain 数），最小化跨块边权总和。
边权 = 依赖扇入扇出数量 × 该边的同步代价。

第一版算法（贪心 + 局部改进）：

1. 按「最大扇入扇出」的算子作为种子，优先放在同一域（比如 MoE 的 GroupGEMM 与它的 expert event）。
2. XCD-aware 重映射：连续 tile 坐标落到同一域，再叠 `GROUP_SIZE_M` 式的 swizzle。
3. 跨域边计数超预算时，把该 event 提升到共享缓存（MALL / 统一 L2）作为汇聚点，或改成静态长队列。

输出：`domain_id` 赋值 + 事件张量的存储类型（普通 VRAM / MALL 常驻 / 跨卡 fine-grained）。

### Pass 3 · 调度模式选择

按 5.5 的公式，逐子图决定 static / dynamic / hybrid。
必须打印可解释的理由，比如：

```
subgraph MoE.groupgemm: dynamic   (S_balance=1.08x, C_cross=2 cross-XCD push, C_queue=low)
subgraph MoE.attn_rs:   static    (cross-device edges=18, C_cross > S_balance)
```

### Pass 4 · 队列分层与任务描述符

- 每域一个局部队列（容量按该域 tile 数上限）
- 全局队列只放跨域 task，条目数上限 = 跨域边数
- 任务描述符压缩成定长记录：`(subgraph_id, tile_coord, task_type)`，避免指针追逐

### Pass 5 · 内存规划与权重预取

- 事件张量、中间张量的 lifetime 与内存类型（见 5.3 的坑）
- 按用户标注为 tile 插入权重预取
- 静态路径生成每 SM 的预排队列并物化到全局内存

---

## 7. L5 后端 lowering 与运行时 ABI

### 7.1 原语 × 架构 lowering 表

| 原语 | NVIDIA | CDNA3/CDNA4 | CDNA2 |
|---|---|---|---|
| `arrive(ev)` agent 域 | `red.global.add` / `atom.global.add` | 全局原子（转出到 IF） | 全局原子 |
| `wait(ev)` | 自旋 + acquire load | 自旋 + acquire load | 同 |
| 组内低代价同步 | cluster + DSMEM | 无（或按目标能力位） | 无 |
| 系统域事件 | `atom.global.system` | 全局原子 + **fine-grained 内存** | 整数原子可以；浮点原子不行 |
| 归约写（通信融合） | `multimem.ld_reduce` | 无对应物 → 原子累加 / L2 驻留归约 | 同 CDNA3 |

**最后一行是我们的最大未知数。** ETC 的 1.40x 通信融合收益建立在 multimem 上，
CDNA 没有对应指令。所以跨卡的 Reduce-Scatter 融合要单独做实验，
做不出正收益就诚实地标成「该硬件不支持此优化」，而不是拿一个凑数的实现去汇报。

### 7.2 生成的内核骨架

```
persistent_kernel(evebuf, queue, desc_table, shape..., domain_map):
    sm = get_sm_id(); dom = domain_of(sm)
    sched = init_scheduler(dom)
    while sched.valid():
        task = sched.next_task()        # static: 预排表; dynamic: pop
        wait_all(task.in_edges)          # 生成自 ETensor 依赖
        execute_tile(task, shape...)
        for ev in task.out_edges:        # notify / push 两选一
            if sched.mode == STATIC: ev.notify()
            else: if ev.notify() == 0: sched.push(ev.consumers)
        sched.advance()
```

### 7.3 运行时 ABI

启动参数只有：

```
event_buf     : 整型张量（所有事件共享一块 buffer）
queue_buf     : 局部队列 + 全局队列
desc_table    : 任务描述符表
shape_scalars : 本轮的符号维度取值
domain_map    : sm_id -> domain_id
```

运行时**没有**任务图遍历器，没有解释执行层。
这是 ETC「lowering 到极小运行时」的延续：调度逻辑编译进内核，运行时只剩两块内存状态。

### 7.4 死锁与活性

- 依赖图无环是编译期保证（4.5 的第一个校验）。
- 数据依赖的 `wait_count` 必须是「上界 + 精确初始化」的组合：初始化写在算出路由的那个 tile 里。
- 持久化内核卡死不会自己报错，会一直超时。开发期必须有：
  1. GPU 侧超时中断（或 host 侧看门狗 kill）
  2. 事件张量 dump，能看到哪个 event 的计数没归零
  3. 可选：把「谁的依赖未满足」编码进 event 的 debug 位

---

## 8. 工作负载覆盖矩阵

| 负载 | 动态性 | 调度选择 | 主要收益来源 | 风险 |
|---|---|---|---|---|
| 低 batch decode（dense） | shape 动态 | static | 打破算子边界，attention 内并行、权重预取 | 算子质量不如手写库时收益被吃掉 |
| MoE（top-k 路由） | 数据依赖 | hybrid：域内 dynamic，域间 static | 域内负载均衡 + GroupGEMM 细粒度流水 | 跨域 push 代价 |
| TP 通信融合（GEMM+RS / AG+GEMM） | 通信抖动 | 前者 dynamic、后者 static | 计算通信重叠 | **CDNA 无 multimem** |
| Prefill（大 batch、计算密集） | shape 动态 | static 或不融合 | 收益有限，主要别退化 | 不融合时也要保证不比基线差 |
| 多模型共置 / 多租户 | 强动态 | 混合（按域切分） | 域级隔离，避免互相拖 | 队列与内存隔离 |
| 非 LLM 的 tile DAG | 视情况 | 由代价模型决定 | 通用性证明 | 缺参考实现 |

「不退化」也是验收项：论文说 ETC 不降低大 batch 性能，我们要把这条写成回归测试。

---

## 9. 验证策略

### 9.1 正确性

1. **差分测试**：每个融合内核实现在 ETX 里必须配一份未融合参考实现（逐算子提交）。
   随机 shape、随机路由、逐元素比对。这是把「融合收益」和「算子质量」分开的唯一办法。
2. **作用域测试**：专门构造跨卡/跨 XCD 的事件，验证真的可见。
   这一条能抓住「系统域原子被静默降级」这类错误。
3. **原子能力断言**：对目标内存类型做一次原子写读回，验证语义成立，不成立则编译期失败。
4. **看门狗**：所有测试带超时，hang 视为失败，并 dump 事件张量。

### 9.2 性能

- 基线三档：逐算子提交（unfused）、unfused-same-op（同算子代码但加全局屏障）、官方库（cuBLAS/RCCL、CK/hipBLASLT）。
  分开报，避免把 tile 质量算进融合收益。
- 每个 PR 跑固定的 shape 集合，超阈值就报警。
- 报数一律带 shape、batch、并行度，不许只报一个加速比。

### 9.3 可移植性回归

同一份 IR，在 gfx90a / gfx942 / gfx950（以及 NVIDIA 侧）上跑同一个小模型：

- 输出必须逐元素一致
- 不许 hang
- 调度决策的日志必须打印能力位差异（证明决策真的读到了机器模型，而不是碰巧一样）

---

## 10. 里程碑与验收

| 阶段 | 交付 | 完成定义（数字） |
|---|---|---|
| 0 · 基线 | MI300X 上 MoE / dense 的逐算子参考实现 + 计时脚手架 + 同步代价标定 | 拿到未融合基线；`T_local`/`T_cross_domain` 两组数字入表 |
| 1 · 单卡闭环 | L2 IR + L5 codegen，静态调度，事件张量按 XCD 分片 | MoE 层对未融合基线 ≥1.05x；逐元素比对通过 |
| 2 · 动态与能力表 | 片上队列、能力表 + 代价表、hybrid 调度 | MoE 上 dynamic ≥ static（对标论文 1024 token 的 1.08 vs 1.04）；打印决策理由 |
| 3 · 第二前端 + 第二硬件 | 接一个非 TVM 前端；同一份 IR 跑 gfx950 或 gfx90a | 换前端/换硬件只改能力表，IR 与 pass 零改动 |
| 4 · 通信融合 | GEMM+RS 在 MI300X 上的融合（无 multimem 的替代路径） | 有正收益就报数，没有就明确列为不支持 |

阶段 1 的 1.05x 是我们自己定的下限：低于这个数说明融合收益被别的开销吃掉了，要先查原因再继续。

---

## 11. 风险与未验证清单

| 风险 | 为什么 | 先测什么 | 失败时的退路 |
|---|---|---|---|
| AMD 无 multimem | 论文的通信融合依赖它 | 阶段 4 前先做单点实验 | 标成不支持，只做计算侧融合 |
| 跨 XCD 同步代价未知 | 4MB/XCD 私有 L2，跨域只有 MALL | 阶段 0 的标定 | 提高同域分片比例，牺牲并行度 |
| cluster/DSMEM 能力按目标变 | LLVM 文档明确说不支持时退化为 agent scope | 逐目标实测并写进 yaml | 一律走全局原子，接受代价 |
| 动态队列争用 | 论文自认中心队列有 contention | 争用曲线标定 | 域级分层队列（6.3 已设计） |
| tile 质量不如手写库 | ETC 自己承认生成的 GEMM tile 不如 cuBLAS | 与手写 tile 做同构对比 | 融合与 tile 生成解耦，允许内联 |
| 静态调度的 shape 兜底太粗 | 论文用「复用下一个更大 shape」 | 形状分布长尾实验 | shape 桶 + 桶内动态微调 |
| 事件张量内存类型写错 | 系统域原子对内存类型有要求 | 作用域测试（9.1.2） | 编译期断言，禁掉非法组合 |

---

## 12. 与 ETC / MPK 的差异

| 维度 | ETC | MPK | ETX（我们） |
|---|---|---|---|
| 抽象 | Event Tensor（一等张量） | SM 级任务图 | 沿用 Event Tensor，加 scope/domain 属性 |
| 调度 | static / dynamic 二选一 | 去中心化 SM 调度 | 静态 / 动态 / **hybrid（域内动态、域间静态）** |
| 硬件 | 仅 B200，8 卡 NVLink | 多卡 NVIDIA | CDNA2/3/4 + Blackwell，能力表驱动 |
| 硬件的处理方式 | IR 层声明无关，实现未验证 | 未强调 | **capability + cost 双表，差异全部数据化** |
| 调度决策 | 用户/论文手选 | 运行时去中心化 | **代价模型自动选，且打印理由** |
| DSL | 声明可接 Triton/CuteDSL，实现是 TVM DSL | 自有前端 | 最小契约（坐标/依赖/资源），浅深两档接入 |
| 内存类型 | 未展开 | 未强调 | 事件张量按 scope 分配，非法组合编译期报错 |

---

## 附录 A · 能力表字段定义

```yaml
arch: gfx942
family: CDNA3
exec_domains:            # 树形，决定分片粒度
  - {name: xcd, count: 8, cus: 38}
wave_size: 64
resources: {lds_kb: 64, vgpr_kb: 512, sgpr_kb: 12.5}
memory:
  l2_per_domain_mb: 4
  l2_total_mb: 32
  shared_cache_mb: 256
  atomic_scope_memory_type: {agent: any, system: fine_grained}
capabilities:
  cluster_launch: unknown
  dsmem: false
  multicast_reduce: false
  fp_atomic_over_fabric: true
  pid_remap_for_locality: true     # 是否支持把 program id 重映射到固定 XCD
costs:                              # 待标定
  t_local_ns: null
  t_cross_domain_ns: null
  t_cross_device_ns: null
  queue_contention_curve: null
```

字段分两类，**不允许混**：`capabilities` 错了是错的代码，`costs` 错了只是慢。

## 附录 B · 标定清单（阶段 0 的具体实验）

1. 同 XCD 内两 CTA 原子计数往返延迟
2. 相邻 XCD / 最远 XCD 的同样测量
3. 事件张量放 MALL 常驻 vs 普通 VRAM 的差异
4. 双卡 P2P 原子 + fence 延迟与带宽
5. N 个 CTA 抢同一队列的吞吐曲线（N 扫描）
6. 静态队列长度 vs straggler 时间（验证静态调度的适用边界）
7. MoE 路由不均程度 vs 动态调度收益（复现论文 1.08 vs 1.04 的条件）

## 附录 C · 术语

| 词 | 含义 |
|---|---|
| task / tile | 一个 CTA 级的工作单元 |
| event | 一组 task 在 SM 粒度上完成的信号 |
| ETensor | 事件组成的多维数组（一等 IR 对象） |
| exec domain | 执行资源层级中的一层：XCD / GCD / die / GPU |
| domain 内 / 跨域 | 同一 exec domain 内 / 跨 exec domain 的同步 |
| capability | 硬件能力位，决定正确性 |
| cost | 同步与放置的代价系数，决定性能 |
| hybrid 调度 | 域内动态、域间静态的混合策略 |

---

## 来源

- Event Tensor / ETC：<https://arxiv.org/abs/2604.13327>（MLSys 2026）
- MPK：<https://arxiv.org/abs/2512.22219>（OSDI'26）
- AMD ROCm GPU 规格表：<https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>
- MI300X 分区文档（SPX/DPX/CPX、NPS）：<https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html>
- MI300/MI350 负载优化（XCD-aware swizzle 实测）：<https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>
- AMD GPU 原子操作支持（按架构与内存 scope）：<https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html>
- LLVM AMDGPU 后端（workgroup cluster launch mode）：<https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html>
