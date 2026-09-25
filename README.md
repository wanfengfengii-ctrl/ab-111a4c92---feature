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

> **安全阀**：搜索始终保持穷举；仅当输入病态（如容差接近同位素间距本身，
> 集合打包搜索空间指数爆炸）导致工作量超过预算时，服务返回 503
> （`SEARCH_SPACE_EXCEEDED`）而非挂起，绝不返回错误裁决。预算可通过环境变量
> `DECONVOLVER_MAX_SEARCH_OPS` 调整（默认 20,000,000 次簇扩展操作）。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/deconvolve` | 解卷积裁决（版本化 JSON 接口） |
| POST | `/api/v1/deconvolve/sensitivity` | 容差敏感性谱扫描（闭区间） |
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

## 容差敏感性谱（`/api/v1/deconvolve/sensitivity`）

峰簇复核前，分析员可提交同一组峰与允许电荷，外加一个**闭区间**容差范围
`tolerance_range = {"lower": …, "upper": …}`（`0 ≤ lower ≤ upper`，均为十进制小数），
一次性取得该范围内裁决随容差放宽而改变的完整敏感性谱，无需多次手工改值比较。

- **临界容差推导**：峰对 `(i, j)` 在电荷 `z` 下的邻接判定为
  `|Δmz·z − 1.003355| ≤ tolerance·z`，因此合法峰簇集合只可能在
  `t = |Δmz·z − 1.003355| / z` 处改变。服务以十进制精度（`Decimal` + 精确有理数）
  从峰差与允许电荷推导**全部**临界容差；对 `z = 3` 这类产生无限循环小数的临界值
  （如 `1/3000000`），内部以 `Fraction` 保持精确，响应中以 `"p/q"` 字符串原样返回。
- **只在端点与临界点重算**：由于邻接判定含等号，裁决在每个半开区间
  `[t_k, t_{k+1})` 上恒定，因此服务仅在范围下端点与落在 `(lower, upper]` 内的
  每个临界点处重新执行既有全局解卷积——**不按固定步长采样**，也无跨请求缓存；
  `evaluation_count == 1 + len(critical_tolerances)` 可直接核验。
- **分段与合并**：结果按容差递增返回若干段，每段给出 `lower`（恒为含端点）、
  `upper`、`upper_inclusive`（仅末段为 `true`，对应闭区间上界）以及该段的完整裁决
  （`verdict`、`objectives`、`clusters`、`unexplained_peaks`、`second_witness`）。
  相邻段若规范化后的裁决、目标值、峰簇、未解释峰及歧义见证完全相同则合并为一段。
- **失败语义**：范围或峰数据非法返回 422（`error.fields[].loc` 可定位），不产生
  部分谱；扫描总工作预算（临界推导 + 各点簇生成 + 全部穷举搜索，由
  `DECONVOLVER_MAX_SENSITIVITY_OPS` 控制，默认 20,000,000）耗尽时返回 503
  `SENSITIVITY_SCAN_FAILED` 并说明失败原因，绝不遗漏临界结论、不返回部分谱。

### 请求示例

```bash
curl -s http://localhost:8000/api/v1/deconvolve/sensitivity \
  -H 'Content-Type: application/json' \
  -d '{
        "peaks": [
          {"mz": "400.000000", "intensity": 5},
          {"mz": "401.003855", "intensity": 7}
        ],
        "charges": [1],
        "tolerance_range": {"lower": "0.0001", "upper": "0.001"}
      }'
```

### 响应示例

```json
{
  "segments": [
    {"lower": "0.0001", "upper": "0.0005", "upper_inclusive": false,
     "verdict": "UNRESOLVED",
     "objectives": {"explained_intensity": 0, "explained_peak_count": 0, "cluster_count": 0},
     "clusters": [], "unexplained_peaks": ["..."], "second_witness": null},
    {"lower": "0.0005", "upper": "0.001", "upper_inclusive": true,
     "verdict": "UNIQUE",
     "objectives": {"explained_intensity": 12, "explained_peak_count": 2, "cluster_count": 1},
     "clusters": [{"charge": 1, "peak_indices": [0, 1], "explained_intensity": 12, "peaks": ["..."]}],
     "unexplained_peaks": [], "second_witness": null}
  ],
  "critical_tolerances": ["0.0005"],
  "evaluation_count": 2,
  "input_summary": {"peak_count": 2, "charges": [1],
    "tolerance_range": {"lower": "0.0001", "upper": "0.001"}, "isotope_spacing": "1.003355"}
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
UNRESOLVED、字典序目标、容差边界、36 峰全量、非法输入 422，以及敏感性谱的
临界点切分、相同裁决合并、非循环小数边界精确性、扫描非法输入等），全部通过时退出码为 0。

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
  sensitivity.py # 容差敏感性谱（临界容差推导 + 分段合并）
tests/           # pytest 单元与接口测试
verify/          # 一次性真实接口验收脚本（compose 的 verify 服务）
Dockerfile
docker-compose.yml
```
