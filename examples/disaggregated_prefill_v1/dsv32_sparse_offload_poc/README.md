# DeepSeek-V3.2 Sparse KV Offload Online P/D 分阶段开发

本目录把同节点 DeepSeek-V3.2 Sparse Host KV Offload 的 Online P/D
启动、预检、验收和证据归档固化为一个入口。目录名保留了历史 `poc`，
当前用途已经是正式功能开发中的分阶段正确性门槛；它仍不是生产部署脚本，
任何新 data plane 通过真实模型和跨机性能验收前都不会替换默认路径。

Mooncake 传输路径、内存类型、真实 NPU 探针和最终 Stop/Go 结论见
[MOONCAKE_TRANSFER_PATHS_2026-08-14.md](MOONCAKE_TRANSFER_PATHS_2026-08-14.md)。

已验证基线：

- 真实 DeepSeek-V3.2 W4A8 权重，运行时 BF16 KV；
- 61 个主模型层，TP8 Prefill + TP8 Decode，Expert Parallel；
- `MooncakeLayerwiseConnector` 逐层 NPU staging；
- framework swapped Full KV、Lightning Indexer Top-K、CANN Gather 和 SFA；
- `max_model_len=4096`，Prompt 覆盖 1818–3618 tokens；
- 超过 `index_topk=2048` 的请求强制执行 8 个 Decode tokens；
- 单请求模式下每个请求完成 8 个 Prefill 和 8 个 Decode rank 生命周期。

## 一次性配置

在服务器容器内执行：

```bash
cd /workspace/w50062541/code/vllm-ascend/examples/disaggregated_prefill_v1/dsv32_sparse_offload_poc
cp --update=none config.example.env config.env
```

根据实际服务器检查并修改 `config.env`。该文件被 Git 忽略，不会提交服务器路径和端口。

配置至少包括：模型路径、Mooncake Python、自定义 OPP、P/D 卡号、API 端口和两个
KV 控制端口基址。共享服务器上的端口和 NPU 所有权是动态状态，
不能写死为“永远可用”。

所有固定 listener 端口必须位于 `/proc/sys/net/ipv4/ip_local_port_range`
之外，或显式加入 `net.ipv4.ip_local_reserved_ports`。否则 Mooncake/MemFabric
启动期间的出站 TCP 连接可能先随机占用这些端口，造成 preflight 时空闲、随后
ZMQ/HCOM bind 随机失败。示例默认使用 22000–22331，并避开 NPU 0–15 的
Mooncake ADXL 20000–21599 端口段。

## 每次启动

先确保 Prefill、Decode 和 Proxy 均未运行，然后执行：

```bash
bash run.sh preflight
```

`preflight` 验证路径、依赖 API、卡号映射、API/KV 控制端口以及 Mooncake ADXL
端口余量。为避免复用过 Prefill/Decode 终端时只看到半组卡，它会在自己的临时
进程中忽略继承的 `ASCEND_RT_VISIBLE_DEVICES`；启动 Prefill/Decode 时仍按
`config.env` 分别设置卡号。它只能验证 NPU 可见性，不能判断共享卡的实际所有权；
还必须确认 `npu-smi info` 中配置的 16 张卡没有其他人的进程。

## Host 传输可行性门槛

当前已验证基线使用 Mooncake Ascend transport 将逐层缓存传入 Decode 普通 NPU
staging。为了评估 Full KV 改走
`P framework swapped Full KV -> D framework swapped Full KV` 是否能减少对
Decode NPU 的干扰，先单独验证当前 Ascend Mooncake wheel 的 TCP Host transport：

```bash
bash run.sh probe-host-transfer
```

该命令不启动模型，分别使用普通 Host 内存和 pinned Host 内存运行两进程传输，
每种内存连续传输 20 个带 guard 的 payload，并检查注册、反注册和进程析构。
测试会在子进程启动前设置 `MC_FORCE_TCP=1`，并要求 Mooncake 原生日志明确出现
`MC_FORCE_TCP is set, using TCP transport only`，避免把 Ascend transport 误认为
Host TCP。该选择机制来自
[Mooncake v0.3.12.post1 TransferEngine 初始化逻辑](https://github.com/kvcache-ai/Mooncake/blob/v0.3.12.post1/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L205-L233)。
脚本只有看到 `2 passed` 才返回成功，`2 skipped` 不算通过；完整输出保存在
`LOG_DIR/dsv32-mooncake-host-transfer-probe.log`，且不得出现 segfault/double free。

这只是新路径的底层 Stop/Go 门槛，不代表 Connector 或真实模型已经切换到 Host
relay。通过后还要依次验证本地 NPU-to-Host、Host-to-NPU-to-swapped/Gather，
以及 Full KV Host transport 与 Indexer NPU transport 的组合生命周期；任一步失败
都不应改动当前默认的 NPU staging 路径。

Host-to-Host 两种内存均通过后，继续验证同一个 worker 进程里的双引擎生命周期：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 bash run.sh probe-hybrid-transfer
```

其中 `8,9` 只是示例；每次运行前必须用 `npu-smi info` 选择两张当时确实空闲的
物理卡。脚本要求显式传入且恰好两张卡，不会清除该映射或擅自触碰物理 0、1 卡；
测试内部看到的逻辑 `npu:0/1` 分别映射到这两张物理卡。

该命令先创建原有 Mooncake Ascend engine，再临时设置 `MC_FORCE_TCP=1` 创建独立的
Host TCP engine，随后立刻恢复进程环境。它依次执行：

```text
Ascend Indexer payload A
  -> TCP pinned-Host Full-KV payloads
  -> Ascend Indexer payload B
```

第二次 Ascend 传输位于 TCP 传输之后，用来确认新增 Host engine 没有破坏原 NPU
engine。发送端和接收端都必须分别出现 Ascend 与 TCP-only 原生日志，使用不同 RPC
端口，通过 payload/guard 校验，并在反注册和析构后正常退出。脚本只有看到 `1 passed`
才返回成功，证据保存在
`LOG_DIR/dsv32-mooncake-hybrid-transfer-probe.log`。

这个测试通过也只确定“双引擎可共存”的实现前提。后续兼容性探针虽然验证了：

```text
P 普通 NPU staging -> P pinned Host relay -> D pinned Host relay
-> D 普通 NPU staging -> D framework swapped Full KV -> Gather
```

但它没有消除 Decode 端 Full KV 经过普通 NPU staging，因此不构成目标传输路径。

双引擎门槛通过后，曾使用下面的兼容性探针验证 Host 数据可以经现有
NPU staging 桥接进入 swapped Full KV：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 bash run.sh probe-host-relay
```

同样要把 `8,9` 换成当时空闲的两张物理卡。该测试只运行 BF16 和真实
DeepSeek-V3.2 Full-KV 维度：NoPE `[4,128,1,512]`、RoPE `[4,128,1,64]`。
它只复制非恒等逻辑映射 `[[2,1]]` 使用的物理块 2、1，并在一个用例内验证：

```text
Prefill NPU staging
  -> Prefill pinned Host
  -> Mooncake TCP Host-to-Host
  -> Decode pinned Host
  -> Decode ordinary NPU staging
  -> Decode framework swapped Full KV
  -> CANN Gather
  -> Decode selected NPU KV
```

测试还会确认物理块 0、3 与 relay guard 未被修改，检查 Host/Ascend 双引擎的
原生日志、内存反注册和进程析构。Decode 侧使用普通 NPU staging 作为 Host buffer
到 swapped Full KV 的本地桥接，与生产 Connector 已验证的 basic-slice persistence
路径保持一致；不直接把 pinned Host 地址写入 swapped/SVM alias。脚本只有看到
`1 passed` 才返回成功，证据保存在
`LOG_DIR/dsv32-mooncake-host-relay-probe.log`。它证明的是兼容性 fallback，
并没有消除 Decode 端 Full KV 经过 NPU staging，因此不是目标 Host 路径，
也不能据此开始性能对比。

用于决定路线的 direct Host Gather 硬件门槛为：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 bash run.sh probe-direct-host-gather
```

其中 `8` 必须替换为当时空闲的一张物理卡。该测试把真实 DeepSeek-V3.2 BF16
Full-KV 维度放在普通 pinned Host Tensor 中，使用非恒等物理块映射 `[[2,1]]`，
直接调用真实 CANN `npu_gather_selection_kv_cache`，不经过 Decode NPU Full-KV
staging，也不使用 swapped Tensor。测试在独立子进程执行，检查 selected KV 数值、
Host 源数据和 guard，并要求子进程正常退出。

该门槛已在当前 CANN/torch_npu/Mooncake 环境的真实 NPU 上运行并失败，关键错误为
`MTE accesses an invalid GM address`。这说明当前公开 Tensor/算子接口不能让 CANN
Gather 直接读取普通 pinned Host Tensor。停止扩展 `host_relay`，保留已验证的
`npu_staging` 基线，并把同一 swapped allocation 的 CPU Host 地址/NPU SVM alias
支持作为下层依赖问题。除非 CANN、torch_npu 或 Mooncake 的相关接口发生变化，
否则不再重复运行该探针或追加同类 bridge 探针。

## MemFabric BM 双视图 Gather G1

目标 Host-to-Host 路径的第一硬件门槛，不再使用普通 pinned Host Tensor。它直接
验证 MemFabric BM 的同一块 DRAM pool 是否可以同时提供：

```text
LOCAL_HOST VA
└── 仅验证地址转换；Host 数据读写通过 BM H2G/G2H API

LOCAL_DEVICE VA
└── 构造 torch NPU Tensor alias，供 NPU copy 和 Gather 使用
```

该探针位于
`tests/ut/distributed/kv_transfer/a3_2/test_memfabric_bm_dual_view_gather_npu.py`，
默认跳过，必须在确认一张空闲 A3 NPU 后显式启用。例如物理 8 卡空闲时：

```bash
cd /workspace/w50062541/code/vllm-ascend

ASCEND_RT_VISIBLE_DEVICES=8 \
VLLM_ASCEND_RUN_MEMFABRIC_BM_GATHER_GATE=1 \
pytest -sv \
  tests/ut/distributed/kv_transfer/a3_2/test_memfabric_bm_dual_view_gather_npu.py::test_memfabric_bm_dual_view_host_full_kv_gather_gate
```

测试完全运行在独立子进程中，依次验证：

1. framework swapped Full KV 可以执行真实 Gather，作为当前环境的正对照；
2. MemFabric BM `HOST` GVA 可转换为 `LOCAL_HOST` 和 `LOCAL_DEVICE` 两个地址；
3. `LOCAL_DEVICE` 地址可按 model runner 的 raw-int8、`view/as_strided` 方式构造成
   BF16 NoPE/RoPE NPU Tensor；
4. `persist_updated_slots` 可以执行 copy ①：普通 NPU staging → BM Host Full KV；
5. 使用不同数据通过 BM `H2G` 写入同一 Host pool 后，真实 Gather 能从 NPU alias
   读取新数据，而不是误读 copy ① 的旧数据；
6. 非目标 block、tensor 间隔和前后 guard 未被修改；
7. 所有 Tensor view 在 BM handle 释放前销毁，进程正常退出且不出现
   segfault/double free。

MemFabric 1.1.2 的 DRAM segment 要求按 1 GiB 对齐，因此探针默认创建 1 GiB
BM pool，但只初始化和检查 Full-KV 布局实际覆盖的区域。该版本返回的
`LOCAL_HOST` VA 不应由 Python `ctypes` 直接解引用；这不是目标数据路径的必要
条件，CPU 侧初始化与校验统一使用 BM `H2G/G2H`，而核心门槛仍是
`LOCAL_DEVICE` alias 能否被 NPU copy 和真实 Gather 读取。

只有输出包含：

```text
MemFabric BM dual-view Host Full-KV -> Gather G1 PASSED
1 passed
```

才判定 G1 为 Go。以下任一情况均为 No-Go 或下层依赖阻塞：

- `gva_to_va(..., LOCAL_DEVICE)` 不存在或返回 0；
- pointer → NPU Tensor 构造接口不可用；
- NPU staging → BM alias 的 `index_copy_` 失败；
- Gather 报 invalid GM address、进程超时或异常退出；
- Gather 仍读到 Host 写入前的旧数据；
- guard 损坏、析构崩溃或 double free。

该测试只使用单 rank BM `SDMA` 验证 allocator、双地址和 Gather 契约，不验证
`HOST_RDMA` 或跨机可见性。G1 通过后才能继续 G2：

```text
P BM Host Full KV
→ MemFabric HOST_RDMA
→ D BM Host Full KV
→ completion/visibility fence
→ D Gather
```

## MemFabric BM 同机双进程 G2a

当前开发环境是一台 16-NPU 服务器，Prefill/Decode 只是用不同 NPU 和独立进程
模拟分离，因此不能宣称跨机 `HOST_RDMA` 已验证。G2 拆成两个门槛：

- G2a：同一物理 Host 上的双进程、双 NPU、两份 BM Host allocation；验证
  P copy①、BM `G2G`、完成通知以及 D `LOCAL_DEVICE` alias 的真实 Gather；
- G2b：未来两台物理 Host 上使用相同数据契约和 `HOST_RDMA`，补测 RNIC、远端
  注册、跨机可见性、fence 和性能。

第一次真实运行已经确认 `HOST_SHM` 不适合作为这个门槛：它能在 `/dev/shm`
建立 CPU 共享映射，但 MemFabric Hybrid 1.1.2 没有为该 segment 发布
`LOCAL_DEVICE` alias，因此两个 rank 的 `gva_to_va(..., LOCAL_DEVICE)` 都返回 0，
程序在 copy①、G2G 和 Gather 之前停止。这是协议与 Gather 内存契约不兼容，
不是 NPU 占用或 `/dev/hugepages` 回退导致的失败。

G2a 改为默认使用 `HOST_TCP`：它仍然传输 P/D 两份 BM Host allocation，且复用
与 `HOST_RDMA` 相同的双视图 DRAM allocator；区别仅是同机功能门槛先走 TCP，
不把它表述为 RDMA 性能结果。测试还会先等待两个 rank 都完成 `join()`，再检查
地址和开始数据阶段，避免先启动的 rank 只看到自己的 group snapshot。

第二次真实运行确认双 rank join、HCOM TCP 建链和 `LOCAL_HOST`/`LOCAL_DEVICE`
双地址均成功，但 MemFabric 的本地 `H2G` 辅助初始化在 `HOST_TCP` 下返回
`507899`。第三次运行进一步确认：默认 `create2(flags=0)` 虽然返回非零且相等的
`LOCAL_HOST`/`LOCAL_DEVICE`，但两端直接解引用 `LOCAL_HOST` 都在 `ctypes.memset`
触发 SIGSEGV。由此可知“地址转换非零”不等于“CPU 可访问”，不能把默认 VMM
地址误称为双视图 Host 内存。

MemFabric Hybrid 1.1.2 源码提供 `SMEM_BM_FLAG_DRAM_MAP_HOST_VA (1 << 9)`：设置后
DRAM allocation 映射 Host VA，并给本地 NPU 添加 READWRITE 权限。G2a 现在显式
通过 `create2(flags=1 << 9)` 请求这个模式，并在 `ctypes` 读写前确认所需区间存在于
`/proc/self/maps` 且可读写。核心 copy①仍由 P NPU staging 写入 P
`LOCAL_DEVICE` alias，copy②仍由 BM `G2G` 执行 P Host→D Host 传输，D 端仍由
真实 Gather 读取 `LOCAL_DEVICE` alias。只有该标志在当前 wheel/driver 上同时满足
CPU Host view、NPU alias、G2G 和 Gather，才能判定目标双视图内存契约成立。

确认两张空闲 NPU 后，例如使用物理 0 卡作为 P、物理 8 卡作为 D：

```bash
cd /workspace/w50062541/code/vllm-ascend

export MF_LIB_DIR=/usr/local/python3.11.10/lib/python3.11/site-packages/memfabric_hybrid/lib
export LD_LIBRARY_PATH="${MF_LIB_DIR}:${LD_LIBRARY_PATH:-}"

python -X faulthandler -m \
  tests.ut.distributed.kv_transfer.a3_2.test_memfabric_bm_same_host_pd_gather_npu \
  --prefill-physical-device 0 \
  --decode-physical-device 8 \
  --start-store-role decode \
  --protocol host_tcp
```

只有输出包含以下内容才判定 G2a 为 Go：

```text
MemFabric BM same-host P -> D Host Full-KV -> Gather G2a PASSED
```

当前服务器已在 MemFabric Hybrid 1.1.2、物理 NPU 0/8、`HOST_TCP` 下通过
`--start-store-role decode` 的 G2a，使 BM rank/store ownership 与 `run.sh` 一致。
结果确认 P/D 两端 `LOCAL_HOST == LOCAL_DEVICE == GVA`、P NPU copy①、BM
`G2G` copy②、D 端真实 Gather、非目标块/guard 和完整清理均通过。这个结果证明
同机双进程功能链路，不是跨机或 RDMA 性能证据。

当前容器不能继续做 `HOST_RDMA` smoke：`/dev/infiniband` 不存在、
`/sys/class/infiniband` 为空，也没有 verbs 设备可供 HCOM 枚举。该缺口只阻塞
G2b/cross-host RDMA 验收，不阻塞下面的真实模型 allocator 集成和同机
`HOST_TCP` data-plane 开发。

`HOST_TCP` 通过后，也可以把协议切成 `host_rdma` 做同机 HCOM 兼容性 smoke。
即使该 smoke 通过，也不能替代 G2b 的真实跨机 `HOST_RDMA` 验收。

## 选择生产传输路径

`config.env` 中的 `SPARSE_KV_TRANSFER_MODE` 控制 Full KV 的逐层 P/D 路径：

```bash
# 已验证基线，也是缺省值
SPARSE_KV_TRANSFER_MODE=npu_staging

# 新的 Host relay 对照路径
SPARSE_KV_TRANSFER_MODE=host_relay

# M2：BM Full-KV Host-to-Host 数据面 + Mooncake Indexer 传输
SPARSE_KV_TRANSFER_MODE=memfabric_bm
```

`npu_staging` 保持原路径：Full KV 与 Indexer 都通过 Mooncake Ascend transport
进入 Decode 普通 NPU staging，再通过本地 NPU→swapped copy 把 Full KV 持久化到
Decode framework swapped Full KV。

`host_relay` 当前是默认关闭的兼容性/诊断模式，使用两套共存的 Mooncake engine：

```text
Full KV: P 普通 NPU staging --本地 NPU→pinned copy--> P pinned Host relay
         --Mooncake TCP--> D pinned Host relay
         --本地 pinned→NPU copy--> D 普通 NPU staging
         --本地 NPU→swapped copy--> D framework swapped Full KV
Indexer: P Indexer NPU KV --Mooncake Ascend NPU→NPU--> D Indexer NPU KV
```

Decode 只有在 Host relay 已桥接到 swapped Full KV、Indexer 也已传完后才确认该层。
这个实现仍让 Full KV 经过 Decode NPU，不是 mentor 所指的最终 Host 路径，当前
不得用它得出性能收益结论。

M1 allocator 门槛已经验证：每个 TP worker 的 Full-KV 底层 allocation 从
`empty_with_swapped_memory` 换成一份 1 GiB MemFabric BM DRAM pool，并将
`LOCAL_DEVICE` view 包装成原布局的 NPU Tensor；真实 61 层模型和 Gather/SFA 均通过。
M1 当时仍由 Mooncake 搬运 Full-KV NPU staging，只证明 allocator/Gather 兼容性。

`memfabric_bm` 现在进入 M2，实现的数据路径为：

```text
P 模型算子
  -> P 普通 NPU staging
  -> 本地 NPU copy
  -> P MemFabric BM dual-view Full KV
  -> MemFabric BM G2G/HOST_TCP
  -> D MemFabric BM dual-view Full KV
  -> Decode visibility fence + layer ACK
  -> Gather selected NPU KV
  -> SFA

P Indexer NPU KV
  -> Mooncake Ascend NPU-to-NPU
  -> D Indexer NPU KV
```

Connector 通过既有 side channel 发布 D 端 BM Full-KV GVA；Mooncake 只注册和传输
Full-KV 之后的 Indexer Tensor，不再注册 P/D Full-KV staging。P 端等待本层
NPU→BM copy 的新 visibility event 后发起同步 G2G；D 端在 ACK 前执行 NPU alias
可见性 fence，因此 Gather 不会在远端 Host 写完成前读取。M2 仍采用逐层同步 ACK，
当前只验证正确性，不得据此声明性能收益或通信计算重叠。

Host 模式会把 4K 配置限制为 33 个物理 block，61 层 BF16 Full-KV 的有效数据约
283 MiB，加上每个 tensor 的 2 MiB 对齐仍落在 1 GiB BM pool 内。若修改层数、
最大长度或 block 数，必须相应增大 `MEMFABRIC_BM_POOL_BYTES`，且保持 1 GiB 整数倍。

M2 同机正确后保留相同 block-offset/fence 契约，在两台真实服务器上把协议切到
`HOST_RDMA` 完成 G2b。HOST_TCP 通过只代表同机功能正确，不代表跨机 RDMA 或性能。

`run.sh` 会主动清除外部 `MC_FORCE_TCP`；不要手工导出它，否则现有
Ascend engine 可能被错误初始化成 TCP。两种模式使用带模式名的独立日志和结果
文件，避免覆盖对照证据。

使用三个终端，按顺序启动：

```bash
# 终端一
bash run.sh decode

# 终端二
bash run.sh prefill

# 终端三；先等待 Prefill ready
bash run.sh proxy
```

`memfabric_bm` 模式下，不要等待 Decode API ready 后才启动 Prefill。Decode 先启动，
看到 `Waiting ... for the MemFabric BM created rendezvous` 后就应
立即在终端二启动 Prefill；两端 BM 建组完成后才会继续到 API ready。这个等待复用
`ASCEND_TRANSFER_TIMEOUT`（当前示例为 600 秒），覆盖 P/D 模型加载耗时不同造成的
启动间隔。每个 TP pair 使用 HCOM 端口段的 offset 0/1 作为 MemFabric 实际监听端口，
offset 2/3 分别作为 `create2` 后和 `join` 后的控制屏障，不承载 KV payload。只有
超时或对端初始化失败才是错误。

`npu_staging` 和 `host_relay` 模式仍可按原基线流程等待 Decode ready 后再启动
Prefill。

可以从第四个终端检查服务：

```bash
bash run.sh ready decode
bash run.sh ready prefill
bash run.sh ready proxy
```

## 正确性验收与证据归档

服务全部 ready 后，在第四个终端执行：

```bash
bash run.sh validate
```

`validate` 包含 Top-K 阈值以下、阈值以上、约 3K、约 3.6K 和请求状态重置
五个用例，并检查每个请求的 P/D rank 生命周期、目标致命错误签名、P/D allocator
初始化，以及 P BM data-plane / D visibility-fence 各 `TP_SIZE` 条。缺少任一 marker
都会直接失败，避免把回退到旧 Full-KV 路径的请求误判为 M2 通过。
计时仅用于发现异常卡顿，不构成性能结论。

通过后，在 Proxy、Prefill、Decode 三个服务终端依次按 `Ctrl+C`。不要在共享服务器
使用会影响同事 Ray/Python 进程的宽泛 `pkill`。
`run.sh` 使用 `tee -i` 让日志采集进程忽略终端的 `SIGINT`；服务进程完成 worker
shutdown 并关闭输出管道后，`tee` 才退出，因此 allocator 的最终释放日志不会在
按下 `Ctrl+C` 时被截断。
`memfabric_bm` 还必须在 P/D 日志中分别看到 8 条
`Released MemFabric BM Full-KV allocator`，且进程正常返回、没有 segfault 或
double free，才算 allocator 生命周期完整通过。停服后执行：

```bash
bash run.sh verify-shutdown
bash run.sh collect
```

`collect` 将完整日志、验收 JSON、Git revision、运行时版本、模型元数据哈希、
停服后的 NPU 状态和传输生命周期 marker 归档到 `OUTPUT_DIR` 下的带时间戳目录。

真实 16 卡 M2 已在 revision `67e01b7b8686e0cee008424dc1e9224bf7bdb6f5`
完成五用例验收：DeepSeek-V3.2 W4A8C8、61 层、TP8 Prefill + TP8 Decode、
`max_model_len=4096`，每个请求 P/D rank 生命周期均为 `8/8`，P BM data-plane 与
D visibility-fence marker 也均为 `8/8`。运行中证据目录为
`/workspace/w50062541/output/dsv32-pd-real61-4k-memfabric_bm-20260821-104138`。
该次停服还确认 16 张 NPU 无残留进程、全部应用/BM 端口恢复空闲且 preflight
重新通过。后续一次停服没有复现原来的 double-free/unmap driver fatal，但 P/D 日志
均缺少 allocator release marker，因此“进程最终消失”和“allocator 按契约显式释放”
仍需分开判断。

## 等待双机期间的单机工程收口

单机不再重复已经通过的正确性套件和 A/B benchmark。当前证据状态是：

| 项目 | 状态 | 结论边界 |
|---|---|---|
| TP8 + TP8 真实模型正确性 | 已闭环 | 同机 `HOST_TCP` |
| MemFabric 30/30 请求稳定性 | 已闭环 | 单请求、同机 |
| `npu_staging` / `memfabric_bm` A/B | 已闭环 | MemFabric TTFT 慢约 6%，不外推双机 |
| NPU→Host copy / Gather trace | 已闭环 | 前台模型执行线程的 NPU scope |
| Connector 后台五阶段耗时 | 代码已补，待一次验收 | Host-observed 聚合时间 |
| allocator 有序释放 | 代码已补，待一次验收 | P/D 各 8 条 release marker |

后台五阶段埋点默认关闭。只在单请求诊断时设置：

```bash
ENABLE_TORCH_PROFILER=false
SPARSE_KV_STAGE_METRICS=true
```

它每个 TP worker 只输出一条聚合 JSON，覆盖 P 侧 visibility wait、MemFabric
Host-to-Host、Mooncake Indexer、ACK round trip，以及 D 侧 visibility fence；不会像
逐层日志那样刷屏。服务 ready 后只跑一个 3018-token 请求：

```bash
cd /workspace/w50062541/code/vllm-ascend/benchmarks/dsv32_pd_transfer
bash run.sh stage-one 3018

cd /workspace/w50062541/code/vllm-ascend/examples/disaggregated_prefill_v1/dsv32_sparse_offload_poc
bash run.sh stage-summary
```

随后正常停止 Proxy、Prefill、Decode，等待两个服务终端都返回 shell，再执行：

```bash
bash run.sh verify-shutdown
bash run.sh collect
```

这一次运行同时承担两个验收目标：`stage-summary` 必须收到 P/D 各 8 个 TP rank
记录，`verify-shutdown` 必须收到 P/D 各 8 个 allocator release marker，且不得出现
driver fatal 或残留 NPU worker。两项通过后，单机工程闭环结束；下一阶段只剩双机
`HOST_RDMA` 正确性、性能和 Decode 干扰验证。

## 当前边界

- 同节点正确性已验证；跨节点仍需独立验证。
- 当前只支持单请求、Eager、`block_size=128`、BF16/FP16 KV。
- 不支持 Prefix Cache、Sparse C8、MTP/Speculative Decode 或 DSA CP/PCP/DCP。
- 逐层 staging 采用同步 ACK；尚未证明通信计算重叠或性能收益。
- `host_relay` 只是默认关闭的兼容性诊断路径，不是性能候选；它使用 TCP 验证
  pinned Host relay 传输，尚未证明跨节点 RDMA/RoCE/UB Host transport，也不使用
  Mooncake Store。
- `memfabric_bm` 当前是默认关闭的 M2 候选：BM `G2G/HOST_TCP` 数据面和真实
  16 卡模型正确性均已在同节点通过；G2b 跨机 `HOST_RDMA` 与性能收益尚未验证。
- 如果端口被 Ray 等共享服务占用，应修改 `config.env` 选择完整空闲端口段，
  不要终止不属于本任务的进程。
