# Rollout Protocol v0 草案（只写文档，不实现）

> 状态：**预研草稿**（TASK-B 提前完工事项）。M6 才启动实现；本文只固化"语义底座 → 传输"的映射思路。
> 语义单一来源是 [docs/rheotrace-spec-v0.md](rheotrace-spec-v0.md)（已冻结）：**协议只做传输，不发明新语义**。

## 1. 设计立场

1. **事件即载荷**：RheoTrace 的事件模型（7 类事件 + 公共信封）直接作为 `StreamEvents` 的载荷 schema。
   本地 trace 文件与网络流是**同一 schema 的两种物理布局**——本地 JSONL，线上 proto。
2. **trace-first 的兼容性承诺**：任何合规 rollout 引擎只要能产出过 validator 的 RheoTrace，
   就能通过一个薄适配器挂上协议；反过来，协议上的会话全程可落盘为 JSONL 供 C 端分析。
3. **版本协商最小化**：`schema_version`（int）+ `format`（string）两个原样字段随首条消息走，不做能力握手矩阵。

## 2. 服务面（proto3 草图）

```protobuf
syntax = "proto3";
package rheo.protocol.v0;

service TrajectoryService {
  rpc SubmitGroups(SubmitGroupsReq) returns (SubmitGroupsAck);          // 提交 rollout 组
  rpc StreamEvents(StreamEventsReq) returns (stream Event);             // 订阅事件流（含轨迹增量）
  rpc SyncWeights(stream WeightChunk) returns (SyncWeightsAck);         // trainer → engine 推权重
  rpc Collect(CollectReq) returns (stream TrajectoryBatch);             // 按预算收割训练数据
  rpc Heartbeat(HeartbeatReq) returns (HeartbeatAck);                   // 存活 + drift 信号上行
}

message SubmitGroupsReq {
  string run_id = 1;
  repeated GroupSpec groups = 2;      // (prompt, G, sampling_params, max_len) —— PLAN L2 Rollout API
  int64  deadline_ns = 3;             // 0 = 无截止
}

message Event {
  int64 ts = 1;                       // 公共信封，字段名与 spec §4.0 一一对应
  string type = 2;                    // eight registered types; unknown → receiver must skip (W06 semantics)
  string run_id = 3;
  oneof payload {                     // 每个 spec 事件类型一条 message；字段名 = JSONL 键名
    RunStart      run_start = 10;
    WeightSync    weight_sync = 11;
    PhaseSpan     phase_span = 12;
    SegmentStart  segment_start = 13;
    SegmentState  segment_state = 14;
    SegmentEnd    segment_end = 15;
    TokenLogprob  token_logprob = 16;
    RunEnd        run_end = 17;
  }
}
```

各 payload message 按规格 §4 字段表逐字段镜像（`lp` → `repeated float`，
`meta` → `google.protobuf.Struct` 保持自由元数据）。**规则**：proto 字段号一旦分配即冻结，
与 spec 的"增量可加、破坏必 bump schema_version"策略一致。

## 3. 关键映射决策（预判 M6 会吵的点）

| 问题 | 草案立场 | 理由 |
|---|---|---|
| logprob 走 float32 还是 double？ | **float**（32 位） | TIS/staleness 对 1e-7 级精度不敏感；带宽省一半。落盘 trace 仍 double——本地保真、线上降密，接受 |
| token_logprob 在线上必发吗？ | 可选（engine 配置） | 训练侧通常自算 new-logprob；线上传 lp 是给外置分析/审计用的 opt-in |
| weight_sync 的权重本体走哪？ | 独立 `SyncWeights` 双向流，**不走 Event 流** | 权重是 GB 级 blob，事件是 KB 级记录；同流会 head-of-line block 事件。事件流里只有 weight_sync 的账本事件（version + 窗口） |
| 乱序与重放 | 事件流内 ts 单调（同 spec E04）；连接重连后用 `last_ack_ts` 续传 | 断线重连不重置版本账本——run_id + version 是全局坐标 |
| 多 worker | `worker` 字段透传（spec §4.3 已有） | 协议不做聚合，聚合是 C 端的事 |
| Heartbeat 里带什么 | 存活 + `drift` 估计（‖ΔW‖/‖W‖、探针 KL，M4 定义）+ 队列水位 | staleness 控制器的输入信号；v0 只留字段位，语义 M4 冻结 |

## 4. 与 Rollout API 的对齐（PLAN L2）

```
river.submit(...)        → SubmitGroups
river.sync_weights(...)  → SyncWeights + Event{weight_sync}
river.collect(...)       → Collect（返回 TrajectoryBatch：segment_end + 关联 token_logprob 的物化视图）
river.abort(...)         → v0 走 Event{segment_state → aborted} 上行；独立 Abort RPC 留给 v1（看使用频率）
```

## 5. 开放问题（M6 前不必答）

1. Arrow 是否替代部分载荷（TrajectoryBatch 的列存形态）——spec §3 已预留 v1 Arrow 布局。
2. 跨引擎联邦：多个 rollout engine 挂同一 trainer 时的 run_id 命名空间。
3. 鉴权/多租户：M6 若只服务单机/单集群，可推迟。
4. `Heartbeat.drift` 的语义冻结依赖 M4 的探针实现（KVT）。

---

*本文档由会话 B 于 2026-09-29 起草；实现启动条件：M1 遥测跑通且 verl 适配器合入（PLAN M6 前置）。*
