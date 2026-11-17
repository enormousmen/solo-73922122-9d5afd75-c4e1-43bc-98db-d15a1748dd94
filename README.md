# Micro-batch Image Classifier

为视觉部署场景（多台相机）提供的**限时图片分类**服务：FastAPI + CPU 版 PyTorch，
单进程微批（micro-batch）推理。提交图片时给出从受理起算的截止毫秒数；
超时或被取消的帧不会产出结果。

## 快速开始

```bash
docker compose up --build
# 服务: http://localhost:8000   交互文档: http://localhost:8000/docs
```

本地开发（无需 Docker）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 提交图片（multipart `file`）+ 表单 `deadline_ms`（从受理起算）。返回 `order_id`。满额返回 **429** |
| GET | `/orders/{order_id}` | 按单号查询状态与结果 |
| POST | `/orders/{order_id}/cancel` | 按单号取消；已完成返回 **409** |
| POST | `/orders/{order_id}/explanation` | 为**已完成**订单申请遮挡解释；返回 **202**。非完成态 **409**，解释队列满 **429**；重复申请幂等 |
| GET | `/orders/{order_id}/explanation` | 查询解释进度/热图；未申请 **404**，订单未完成 **409** |
| GET | `/stats` | 容量/状态计数（演示与观测用） |
| GET | `/health` | 存活探针 |

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000"
# {"order_id":"7f6770829f36","status":"queued",...}

curl http://localhost:8000/orders/7f6770829f36
# {"status":"completed","result":{"predicted_class":1,"scores":[...10 个分数...]},...}

curl -X POST http://localhost:8000/orders/7f6770829f36/cancel
```

单号状态机：`queued → processing → completed`；`queued/processing → cancelled|expired`。
`result.scores` 为 10 类 softmax 分数，与提交单号一一对应。

## 遮挡解释 API（已完成订单）

排查“相机图片错分”时，用遮挡法定位对原判影响最大的区域：

```bash
curl -X POST http://localhost:8000/orders/7f6770829f36/explanation
# 202 {"status":"queued","grid":[4,4],"progress":{"cells_done":0,"cells_total":16},...}

curl http://localhost:8000/orders/7f6770829f36/explanation
# {"status":"completed","predicted_class":1,"original_prob":0.1053,
#  "heatmap":[[...4 个...],[...],[...],[...]],
#  "progress":{"cells_done":16,"cells_total":16},"error":null}
```

- 把该订单原来的 64×64 输入切成 **4×4 = 16 格**，逐格把该格归一化像素置 0
  （即原始灰度中点 127/128），重新推理。
- `heatmap[r][c] = 原预测类别的原始概率 − 遮挡该格后同一类别的概率`；
  **负差值照实保留**，不做截断。`original_prob` 取自原订单结果，两者口径一致。
- 解释与原帧推理**共用单计算通道**（同一单线程执行器，绝不同时占用模型）；
  每轮最多计算 **4 张**遮挡图，两轮之间**优先处理已受理的原帧**，因此 16 格分 4 轮完成。
- 部分完成（`queued`/`processing`）时 `heatmap` 恒为 `null`；任一格出错则该解释标记
  `failed` 并返回 `error`，**不返回部分热图**。
- 解释只读取已完成订单：**不改动原订单状态、分数、唯一约束**；原订单该取消仍返回 409。
  非完成态（排队/计算中/取消/过期）申请返回 **409**，未知单号 **404**。
- 解释与图片容量分开计量：解释 FIFO 最多 `MAX_CAPACITY`（16）个待处理任务，满额 **429**。
  解释没有截止时间，不影响原帧的微批/截止/取消规则；慢解释期间提交、查询、取消仍即时响应。
- 解释完成后再次 POST 幂等返回同一结果；`failed` 可再次 POST 重试。
- `/stats` 额外给出 `explanations_queued`、`explanations_active`、
  `explain_rounds_started`、`last_explain_round_size` 便于观测。

## 规则实现（与需求逐条对应）

- **容量 16**：排队中 + 计算中合计最多 16 张，满额提交返回 429。
- **单批 ≤ 4，等批 ≤ 50ms**：模型空闲时最多等 50ms 凑批，凑满 4 张立即开批。
- **最多一批实际推理**：单个 worker 协程 + 单线程执行器，任何时刻只有一批在做前向计算。
- **排队取消/到期**：从队列移除、释放容量，**不进入模型**（`batch_id` 始终为空）。
- **入批后取消/到期**：立即标记（查询即刻可见）、不供结果；**容量占用到该批计算实际结束**，
  期间不会因此另开一批（worker 严格串行）。
- **批内其他图片正常完成**，分数对应原单；批推理与逐张推理误差 ≤ 1e-5（实测约 7.5e-9）。
- **迟到结果不恢复取消/过期单**：结果回写只针对仍处于 `processing` 且未过期的单。
- **慢推理不挡 API**：阻塞式前向计算放在独立线程，事件循环始终可用。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_CAPACITY` | 16 | 排队+计算中最大图片数 |
| `BATCH_SIZE` | 4 | 单批最大图片数 |
| `BATCH_WAIT_MS` | 50 | 空闲模型凑批等待上限 |
| `INFERENCE_DELAY_MS` | 0 | **演示用**：在推理线程内注入的人为延迟 |
| `MODEL_PATH` | `app/model.pt` | TorchScript 模型路径 |

慢推理演示：`docker compose up` 前把 `docker-compose.yml` 里的 `INFERENCE_DELAY_MS`
改为 `2000`，推理约 2 秒/批，期间提交/查询/取消仍在毫秒级响应（可用 `/stats` 观察）。

## 模型与样图

- `app/model.pt`：固定权重 TorchScript 模型（小型 CNN，10 类，输入 64×64 RGB）。
  由 `scripts/make_model.py` 以固定种子生成，权重 SHA-256 指纹见脚本输出，可复现。
- `samples/`：4 张确定性生成的样图（渐变/圆环/条纹/噪声）。

重新生成：

```bash
python scripts/make_model.py --out app/model.pt
python scripts/make_samples.py --out samples
```

## 测试

```bash
.venv/bin/pytest tests/ -q
```

覆盖：单张完成、批/单张分数一致性（≤1e-5）、凑批窗口、容量 429、排队取消、
入批取消（立即标记+容量仍占用+不另开批）、完成单取消冲突、排队过期、
计算中过期、迟到结果不复活、慢推理下 API 延迟；
遮挡解释：热图与独立参考实现逐格一致（≤1e-5、负值保留）、与原订单结果口径一致、
每轮 ≤4 图且 16 格分 4 轮、仅完成单可申请（排队/计算中/取消/过期均 409、未知 404）、
计算中不返回部分热图、失败不返回热图且不改原订单并可重试、帧优先、慢解释下 API 延迟、
解释队列容量与幂等。

## 已知限制

- 订单记录保存在内存中，重启即丢失；未做持久化与历史订单清理（演示规模）。
- 解释任务同样只存内存；解释队列上限复用 `MAX_CAPACITY`，没有单独的配置项。
- 单进程单 worker；多副本扩展需外加负载均衡与共享状态层，不在本交付范围。
