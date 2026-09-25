# Isotope Peak Deconvolution Service

供高分辨质谱（HRMS）实验室复核重叠同位素峰的**纯后端服务**：Python 3.13 + FastAPI，
无前端。分析员提交一组按质荷比严格递增的峰，服务执行**确定性、穷举式**解卷积，
返回峰簇划分与裁决（`UNIQUE` / `AMBIGUOUS` / `UNRESOLVED`）。

## 问题定义

- 输入：2–36 个峰（`mz` 为正的十进制小数、严格递增；`intensity` 为正整数）、
  允许电荷集合 `charges`（正整数、互不重复）、十进制容差 `tolerance`（≥ 0）。
- 峰簇：2–6 个峰、同一电荷 `z`，相邻质荷比之差与 `1.003355 / z` 的偏差不超过容差
  （内部以 `|Δmz·z − 1.003355| ≤ tolerance·z` 精确判定，无浮点误差）。
- 每个峰至多属于一个峰簇。
- 求解器**完整搜索所有合法峰簇组合**（基于位掩码的精确动态规划，非贪心、
  非"逐峰就近"、非"先选最强候选"），按字典序依次优化：
  1. 最大化已解释总强度；
  2. 最大化已解释峰数；
  3. 最小化峰簇数。
- 裁决：
  - `UNIQUE`：最优组合唯一；
  - `AMBIGUOUS`：三项目标完全相同的最优组合不止一个，响应附带一份不同的
    见证（`second_witness`）；
  - `UNRESOLVED`：不存在任何合法峰簇。
- 非法输入返回 422，错误体给出可定位字段（`error.fields[].loc`），且不产生裁决。

### 容差敏感性谱

复核峰簇前，分析员可提交同一组峰、允许电荷以及一个**闭区间**容差范围
`tolerance_range = [lower, upper]`（`0 ≤ lower ≤ upper`），一次性取得该范围内
裁决随容差放宽而改变的完整敏感性谱，无需多次手工改值比较：

- 服务以十进制精度从**相邻峰差与允许电荷**推导全部可能改变合法峰簇集合的
  **临界容差**：对每对峰 `i < j` 与每个电荷 `z`，临界值为
  `|Δmz·z − 1.003355| / z`（间距判定为含等号，故峰簇恰在其临界容差处"诞生"）。
- 仅在**范围端点与这些临界点**重算既有全局解卷积——绝不按固定步长采样，
  也不复用先前请求的结果；相邻求值点之间合法峰簇集合恒定，故裁决恒定。
- 相邻区间若规范化后的裁决、目标值、峰簇、未解释峰及歧义见证完全相同则合并。
- 返回按容差递增的段序列，每段给出 `lower` / `upper` 边界、完整裁决
  （`verdict`、目标值、峰簇、未解释峰、歧义见证）以及 `upper_inclusive`
  （仅末段为 `true`，因为扫描区间在上界闭合；其余段为 `[lower, upper)`）。
- 边界以十进制字符串返回：可精确表示的临界值原样输出；无有限十进制展开的
  临界值（如 `z=3` 时的 `0.1/3`）内部仍以精确有理数参与判定，仅显示值
  舍入到 40 位有效数字。
- 任一求值点耗尽搜索预算时返回 503（`SENSITIVITY_SCAN_EXCEEDED`）并明确说明
  扫描失败原因，绝不返回遗漏临界结论的部分谱；范围或峰数据非法时返回 422，
  同样不产生任何部分谱。

> **安全阀**：搜索始终保持穷举；仅当输入病态（如容差接近同位素间距本身，
> 集合打包搜索空间指数爆炸）导致工作量超过预算时，服务返回 503
> （`SEARCH_SPACE_EXCEEDED`）而非挂起，绝不返回错误裁决。预算可通过环境变量
> `DECONVOLVER_MAX_SEARCH_OPS` 调整（默认 20,000,000 次簇扩展操作）。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/deconvolve` | 解卷积裁决（版本化 JSON 接口） |
| POST | `/api/v1/sensitivity-spectrum` | 闭区间容差范围内的精确敏感性谱 |
| GET | `/health` | 健康检查 |
| GET | `/docs` | OpenAPI 交互文档 |

### 请求示例

```bash
curl -s http://localhost:8000/api/v1/deconvolve \
  -H 'Content-Type: application/json' \
  -d '{
        "peaks": [
          {"mz": "500.000000", "intensity": 1000},
          {"mz": "501.003355", "intensity": 800},
          {"mz": "502.006710", "intensity": 600}
        ],
        "charges": [1],
        "tolerance": "0.0005"
      }'
```

### 响应示例（节选）

```json
{
  "verdict": "UNIQUE",
  "objectives": {"explained_intensity": 2400, "explained_peak_count": 3, "cluster_count": 1},
  "clusters": [
    {"charge": 1, "peak_indices": [0, 1, 2], "explained_intensity": 2400,
     "peaks": [{"index": 0, "mz": "500.000000", "intensity": 1000}, "..."]}
  ],
  "unexplained_peaks": [],
  "second_witness": null,
  "input_summary": {"peak_count": 3, "charges": [1], "tolerance": "0.0005", "isotope_spacing": "1.003355"}
}
```

`clusters` 按（首峰 m/z、电荷、峰下标）规范排序；`mz` 以字符串原样返回以保持十进制精度。

### 错误响应（422）

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input; no deconvolution verdict was produced.",
    "fields": [{"loc": "peaks.1.intensity", "message": "Input should be greater than 0", "type": "greater_than"}]
  }
}
```

### 敏感性谱请求 / 响应示例

```bash
curl -s http://localhost:8000/api/v1/sensitivity-spectrum \
  -H 'Content-Type: application/json' \
  -d '{
        "peaks": [
          {"mz": "300.0", "intensity": 100},
          {"mz": "300.6", "intensity": 200},
          {"mz": "301.0", "intensity": 200}
        ],
        "charges": [1],
        "tolerance_range": {"lower": "0", "upper": "0.7"}
      }'
```

```json
{
  "segments": [
    {"lower": "0", "upper": "0.003355", "upper_inclusive": false,
     "adjudication": {"verdict": "UNRESOLVED", "...": "..."}},
    {"lower": "0.003355", "upper": "0.403355", "upper_inclusive": false,
     "adjudication": {"verdict": "UNIQUE", "...": "..."}},
    {"lower": "0.403355", "upper": "0.603355", "upper_inclusive": false,
     "adjudication": {"verdict": "AMBIGUOUS", "...": "..."}},
    {"lower": "0.603355", "upper": "0.7", "upper_inclusive": true,
     "adjudication": {"verdict": "UNIQUE", "...": "..."}}
  ],
  "evaluation_point_count": 5,
  "critical_point_count": 3,
  "input_summary": {"peak_count": 3, "charges": [1],
    "tolerance_range": {"lower": "0", "upper": "0.7"}, "isotope_spacing": "1.003355"}
}
```

## 快速开始（Docker）

```bash
# 构建并启动 API（宿主机端口默认 8000，可用 API_PORT 覆盖）
docker compose up --build

# 自定义宿主机端口
API_PORT=9000 docker compose up --build

# 一次性运行真实接口验收（verify 服务，依赖 api 健康检查后自动执行）
docker compose run --rm verify
# 或：docker compose --profile verify up --abort-on-container-exit
```

`verify` 服务对运行中的真实 API 执行全部验收场景（UNIQUE / AMBIGUOUS /
UNRESOLVED、字典序目标、容差边界、36 峰全量、非法输入 422，以及真实接口的
敏感性谱扫描：临界容差分段、相邻合并、闭区间上界包含、非终止临界值等），
全部通过时退出码为 0。

## 本地开发

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.main:app --reload --port 8000   # 启动服务
pytest                                       # 单元 / API 测试
python verify/verify_acceptance.py           # 对本机实例跑验收（API_BASE_URL 可覆盖）
```

## 目录结构

```
app/
  main.py        # FastAPI 应用、路由、错误处理
  schemas.py     # 请求/响应模型（Pydantic 校验）
  solver.py      # 穷举式精确求解器（Decimal/Fraction 精确运算）
  sensitivity.py # 临界容差推导与敏感性谱扫描（精确有理数）
tests/           # pytest 单元与接口测试
verify/          # 一次性真实接口验收脚本（compose 的 verify 服务）
Dockerfile
docker-compose.yml
```
