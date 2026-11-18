# 规范条款精准翻译：受控对照实验

本目录包含一套可复现的英文规范条款翻译实验。题库为原创模拟标准语句，开发集与独立测试集按语义家族隔离；固定中文锚点在模型调用前冻结。四种主流程（Direct、独立片段、共享三候选池后的聚合和选择）都使用正确术语卡；Direct 另设无卡和检索卡两个信息对照。

## 离线检查

Python 3.10+。runner、离线测试和 CSV/SVG 汇总只用标准库；语义结果 PNG 绘制需要 Matplotlib 与 NumPy。

```powershell
python -m unittest discover -s tests -v
python run_experiment.py --help
python analyze_results.py --help
```

离线检查不需要 API 凭据，也不会发起网络请求。

## 运行实验

先在启动 Python 的同一进程环境中设置 `CASDAO_API_KEY`。程序只从该环境变量读取密钥，不提供命令行、配置文件或 `.env` 凭据入口；请求头、密钥和请求正文不会写入日志或缓存。

```powershell
$env:CASDAO_API_KEY = "<由运行者在本地环境中提供>"
python run_experiment.py --split all --workers 3 --output results/experiment.jsonl
```

默认端点为 `https://maasapi.casdao.com/v1/chat/completions`，主模型为 `deepseek-v4.1-flash`。也可用 `--split dev` 或 `--split test` 单独运行；`--workers` 默认 3，最大 4。程序以逐请求 SQLite 缓存和逐条件检查点支持中断恢复。要续跑同一批实验，重复相同命令即可；成功条件会跳过，失败条件会重试。API 调用没有内置总量上限。

## 结果与盲审

每题运行 6 个最终译文条件：四个正确卡主流程、Direct 无卡和 Direct 检索卡。候选池只生成一次，聚合和选择共同使用，因此 40 题预计约 400 次模型请求（开发集约 100 次、测试集约 300 次；失败重试或缓存命中会改变实际数值）。

完成模型运行后执行：

```powershell
python analyze_results.py --input results/experiment.jsonl --out-dir results/analysis
```

生成汇总表、逐条件覆盖与调用成本、自动规则检查、匿名盲审 JSONL、匿名 ID 到实验条件的映射，以及按源句组织的分歧索引。`results/analysis/blind_review.jsonl` 只含测试集，另有 `blind_review_dev.jsonl`；每行对应一个去重后的英文原句、冻结中文锚点和唯一模型译文，字段不包含方法、卡片、数据家族或来源。主流程译文、两个 Direct 信息对照和主流程三候选均纳入去重；相同源译对只分配一个盲审 ID。`review_mapping.jsonl` 用于归档后映射回条件，不提供给评审者。`.blind_salt` 是固定的非鉴权 ID 随机种子，应与盲审文件和评分一同归档以保持既有 ID。GPT-6.2 Astra 主代理根据 `review_rubric.md` 完成语义盲评。自动规则输出只报告词面代理检查，不是语义判分；runner 不生成自动或模型语义标签。

API 原始结果、CSV/JSON 汇总、JSONL 盲审/评分/映射、图表及绘图代码均为复现与报告材料，应归档。仅请求缓存、SQLite 检查点、Python 字节码和 `.tmp` 临时文件由 `.gitignore` 排除。

已有评分的重算流程：先保留 `results/analysis/astra_ratings.jsonl`、`review_mapping.jsonl` 和 `.blind_salt`，然后可重跑汇总与盲审生成器，最后运行：

```powershell
python analyze_semantics.py --ratings results/analysis/astra_ratings.jsonl `
  --results results/experiment.jsonl --analysis-dir results/analysis `
  --out-dir results/analysis/semantic
```

分析器核对评分 ID 与冻结盲审 ID 集合完全一致，并拒绝重复 ID 或未知标签；它不改写评分文件。

检查通过率、自动规则结果、语义判断和分歧是不同证据。不要把自动检查代理写成语义正确率。候选池在两种后处理中共享，报告方法级逻辑成本与实际去重后的网络成本时须说明这一点。结果文件只在相应调用成功后生成，不应把缓存命中数解释为新模型生成次数。

## 文件布局

- `data/items.jsonl`：40 条冻结条款与中文锚点，8 个语义家族；2 个开发家族（10 条）和 6 个测试家族（30 条）。
- `data/cards.json`：固定术语卡、检索术语卡与检索触发词。
- `data/source_notes.md`：RFC 来源核验与原创数据说明。
- `protocol_freeze.md`：预注册式冻结设计、处理流程、指标和停机规则。
- `review_rubric.md`：GPT-6.2 Astra 主代理逐条盲评规程。
- `run_experiment.py`：OpenAI Chat Completions 兼容客户端和实验 runner。
- `analyze_results.py`：汇总、盲审去重、自动规则检查和绘图。
- `analyze_semantics.py`：读取主代理评分并回填条件、误差、候选覆盖与成本分析。
- `tests/`：离线数据、方法、缓存和去标识测试。
- `results/`：运行生成的结果、盲审材料、主代理评分、解盲映射、固定 ID 随机种子、图表与缓存。归档实验时保留映射和盐以复算评分对应关系；如需另行分发匿名盲审材料，只分发盲审 JSONL，不连同解盲映射。

