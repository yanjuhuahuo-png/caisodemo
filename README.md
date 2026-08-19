# CAISO 价差交易决策辅助 · V0.3.1.2（Demo Freeze）

预测 CAISO 日前（DA）与实时（RTPD）价差 `Return = DA − RTPD` 的方向与幅度，辅助 **SELL_DA / BUY_DA / NO_TRADE** 交易决策。V0.3.1 把 V0.2 的白盒决策链封装成**浏览器可用的 Decision Workspace + 自然语言 Ask Trading Agent**：选案例 → RUN → LOCK → REVEAL → ASK → BRIEF，全流程可审计、可解释、不穿越。V0.3.1.3（Demo Freeze Polish）统一了跨平台哈希、available-at-only Time Gate、Golden Case E 稳定证据演示与版本号。

> **诚实标注（请先读）**
> - **MODEL SIGNAL IS EXPERIMENTAL / CURRENT ALPHA = WEAK** —— 模型信号是实验性的，当前能力很弱。
> - **Signal / Strategy MVP，不是已验证盈利系统** —— 这是演示/研究系统，不是能保证赚钱的交易软件。
> - **LLM 不决定方向** —— 交易方向一律由白盒 DecisionService + 6 个结构化 Tool 程序化决定；LLM 只做解释，且不能修改/覆盖工具返回的数字与最终建议（程序化完整性守卫拦截）。
> - **决策路径无 MOCK** —— 特征、证据、历史案例全部真实且 as-of；无真实证据时诚实显示 `NO ELIGIBLE EXTERNAL EVIDENCE`，绝不编造。
> - **DEMO 证据是真实历史快照** —— DEMO MODE 用 `demo_artifacts/evidence_demo.json`（真实历史 GFS 18Z 快照，可重复、不依赖现场网络），页面明确标注 `EVIDENCE MODE: HISTORICAL SNAPSHOT`，绝不写成实时。

---

## 1. 系统是什么

加州电力市场里"电力"分两种方式买卖：**DA**（提前一天约定，10:00 前报价、约 13:00 出结果）和 **RTPD**（当天实时逐小时结算）。两者之差即价差 `Return = DA − RTPD`。

系统做**虚拟交易**（Convergence Bidding，1 MWh/仓）：
- 预期 `Return > 0`（DA 高于 RTPD）→ **SELL_DA**，赚 `DA − RTPD`；
- 预期 `Return < 0` → **BUY_DA**，赚 `RTPD − DA`；
- 判断不出 / 风控拒绝 → **NO_TRADE**（PnL = 0）。

一笔决策的完整链路：`10:00 PT cutoff → as-of 特征 → 模型预测 → Agent Evidence（Time Gate 过滤）→ 历史相似案例（as-of）→ Risk Gate → 白盒 Rule Engine → LOCK → REVEAL Actual → PnL → Post-trade Review → Audit`。

## 2. 一键启动

前置：Python 3 + 已生成数据 artifact（`code/data/canonical.parquet`、`predictions_v2.csv`、`stage3/risk_features.parquet`、`agent/case_library/cases*.json`）。

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 启动前自检（可选；缺 artifact 时给出重建指引）
python prepare_mvp.py

# 3) 启动 Web（默认 http://127.0.0.1:5000）
python mvp_web.py
#    不取外部 GFS 证据、纯本地演示：
python mvp_web.py --offline
```

打开浏览器访问 `http://127.0.0.1:5000`。

> 若 `pip install -r requirements.txt` 后无法启动，先跑 `python prepare_mvp.py`，它会逐项检查 artifact 与关键依赖，缺失时给出具体重建命令。
> 依赖清单见 `requirements.txt`；LLM 官方 SDK（openai/anthropic）为可选 —— 不装时 Ask Agent 走 httpx/requests 直连或诚实降级。

**数据模式（FULL / DEMO）**：`prepare_mvp.py` 自动探测——
- **FULL**：完整数据 artifacts 存在（`code/data/*`）。
- **DEMO**：完整数据缺失（clean clone），但 `demo_artifacts/` 存在（真实历史最小切片，只覆盖 5 个
  Golden Cases）。DEMO **≠ MOCK**：DEMO 是真实历史记录的子集，可给出真实推荐；MOCK 永不参与真实推荐。
- 环境变量可覆盖：`MVP_DATA_MODE=demo|full`（旧别名 `DATA_MODE` 兼容）。
- 在完整数据机上运行 `python build_demo_artifacts.py --check` 可重新生成切片并校验
  FULL vs DEMO 一致性（`docs/v0311_consistency.md`，5/5 PASS）。

## 3. Demo 怎么操作

首页是 **Decision Workspace**（交易决策工作台），完整闭环 6 步：

| 步骤 | 操作 | 说明 |
|---|---|---|
| ① **选案例** | 顶部"黄金案例"（B/C1/C2/D/E）或自定义 `decision_date × node × hour`，选证据模式（真实证据 / 离线静态） | 合法决策日 ≈ `2026-06-01 ~ 2026-08-04`（test 窗口）；三节点 `CONTROLX_1_N001`、`SNLNDRO_1_N001`（ZP26）、`ELCAJNGT_7_N001`（SP15 冷启动）。**证据模式**：DEMO MODE 用真实历史 GFS 快照（`EVIDENCE MODE: HISTORICAL SNAPSHOT`，可重复、不依赖现场网络）；FULL MODE 用实时 GFS（`LIVE`）；离线静态（`NONE`） |
| ② **RUN** | 点 RUN DECISION | 服务端跑完整白盒决策链，返回结构化决策对象（模型/证据/案例/风控/规则/最终建议） |
| ③ **LOCK** | 点 LOCK DECISION | **锁定决策**。锁定前系统绝不展示任何 actual/outcome |
| ④ **REVEAL** | 点 REVEAL ACTUAL OUTCOME（仅锁定后可点） | 揭晓真实 DA/RTPD/Return，算 PnL（1 MWh）与复盘分类 |
| ⑤ **ASK** | Ask Trading Agent 面板输入自然语言问题 | LLM 经 6 个 Tool 取事实回答并展示 **Agent Trace**；无 Key 时诚实显示 `LLM NOT CONFIGURED` |
| ⑥ **BRIEF** | GENERATE DAILY BRIEF | 扫描指定日全部已生成决策，输出 BUY/SELL/NO_TRADE 汇总、Top 机会、Top 风险 |

**Golden Case E（稳定演示 "防信息穿越"）**：DEMO MODE 下选择案例 E（2026-07-08 CONTROLX H2），页面显示一条**真实历史 GFS 18Z 预报快照**（`EVIDENCE MODE: HISTORICAL SNAPSHOT`），明确展示 Forecast Run / Initialization Time / Published At / Available At / Decision Cutoff → `Initialization Time 晚于 Cutoff` → `Decision Eligible: NO` → `Reason: INITIALIZATION_AFTER_CUTOFF` → `NOT USED`。该证据的**真实可用时刻无法证明**（`Available At: UNKNOWN / NOT PROVEN`，不伪造 init+delay），但仅凭初始化晚于截止即可确定不可能参与当时决策。该证据**不进入** Risk Gate / Rule Engine / 最终建议——交易结果与无证据时完全一致（不穿越、可重复、可现场审计）。

## 4. 每块内容是什么意思（速查表）

| 屏幕上/决策对象里的英文 | 中文意思 | 备注 |
|---|---|---|
| Decision Cutoff | 决策截止 = 当天 **10:00 PT**（DAM 官方 bid cutoff） | 系统的"时间红线" |
| as-of / decision_eligible | "截至该时点可获得 / 是否合规用于决策" | 防穿越核心概念 |
| POST-DECISION / NOT USED | 晚于 cutoff 的证据，只进复盘 | 说明系统不偷看未来 |
| expected_return | 预期价差（DA−RTPD，$/MWh） | 决定方向 |
| model_signal_strength | 模型信号强度（**非概率**） | 仅供参考，不能当置信度 |
| uncertainty | 不确定度 | 越大越没把握 |
| Risk Gate REJECT | 风控拒绝入场（一票否决） | PASS / WARNING / REJECT |
| BUY_DA / SELL_DA / NO_TRADE | 买入日前 / 卖出日前 / 不交易 | 最终动作 |
| PnL | 盈亏（$ / MWh，1 MWh 仓） | 事后结算 |
| reason_code | 规则命中的编号 | 审计用（页面有业务翻译） |
| Agent Trace | LLM 一次问答的工具调用流水 | 谁调了什么工具、返回了什么 |
| availability_basis | 特征可用性依据（STATIC/STRUCTURAL_LAG/ASSUMED_AVAILABLE/…） | 展示口径 == Time Gate 判定口径 |
| EVIDENCE MODE | 证据来源模式：HISTORICAL SNAPSHOT（真实历史 GFS 快照）/ LIVE（实时 GFS）/ NONE（离线） | DEMO 用历史快照可重复、不依赖现场网络；绝不把快照写成实时 |
| Market Rule Version | 单笔决策的市场规则版本（date-aware，按 target_date 判定） | 2026-05-01 边界 → `POST_DAME_EDAM_2026`；展示值 == DecisionSnapshot.context |

## 5. Ask Trading Agent（LLM Copilot）

页面上方 **Ask Agent** 面板支持自然语言追问。实现：`code/llm_copilot.py`，通过 6 个结构化 Tool 获取事实：

1. `get_decision(decision_date, node, hour)` — 模型输出 / 风控 / 规则引擎 / 最终建议
2. `get_feature_explanation(decision_id)` — Top 特征 + 值 + z 贡献 + 来源 + 可用性
3. `get_evidence(decision_id)` — eligible / rejected 证据 + 来源 + 发布时间 + 隔离原因
4. `get_similar_cases(decision_id)` — Top 3 相似案例 + `case_available_at` 证明 as-of
5. `get_data_provenance(decision_id)` — 特征来源 / 原始文件 / 可用性 / 是否 MOCK
6. `get_post_trade_review(decision_id)` — 仅 `outcome_revealed=True` 可调用，否则 `OUTCOME_NOT_REVEALED`

预设问题自动选工具（如"为什么不卖"→ get_decision；"类似案例"→ get_similar_cases）；非预设问题走 LLM tool-calling / router。**铁律**：LLM 只解释不决策，工具返回的数字与最终建议不可被 LLM 覆盖（程序化完整性守卫，违反即拦截并记入 Trace）。

**无 API Key 时**：核心决策流程照常运行；Ask 面板诚实显示 `LLM NOT CONFIGURED`，不会编造回答。

### 5.1 结论可信度核验（V0.4.2）

页面新增 **"结论可信度核验"** 卡片（决策卡"为什么这样建议"下方）：点击后回答 **"智能体给出的结论是否可信？为什么？"**

- **核验只依据真实数据事实**：7 项运行时审计（feature/evidence 时间门槛、MOCK 隔离、Case as-of、结果泄漏、证据可用性/血缘）+ provenance + 证据 Time Gate 结果，全部程序化采集（`code/decision_service.py` 新工具 `verify_conclusion`，第 7 个 Tool）。
- **可信度等级由确定性门槛锁定**：`TRUSTWORTHY`（审计全 PASS）/ `CAUTION`（数据真实但有提醒项）/ `NOT_TRUSTWORTHY`（审计 FAIL，路径含 MOCK/泄漏/穿越）。LLM 只能解释理由，**不可改判**（输出与门槛冲突即拦截，回退程序化理由）。
- **数据可信才输出结论**：`TRUSTWORTHY` 时明确给出可采信的最终建议（BUY_DA/SELL_DA/NO_TRADE）与理由；`NOT_TRUSTWORTHY` 时明确"结论不可采信、不得据此交易"。
- LLM 缺席（无 Key）时自动使用**程序化理由**（诚实降级，等级判定不变）。
- 入口：决策页核验按钮；Ask Agent 问"这结论可信吗 / 数据可信吗 / 帮我核验"；API `POST /api/verify/<decision_id>`。

### 5.2 未来交易日预测（V0.4.3 · LLM 推理预测，实验性）

页面 `/forecast`（决策工作台首页链接）：**自选 2026-08-05 ~ 2026-08-09 中的任意一天 + 节点**，预测该日电价并给出 **"多少钱买入 / 多少钱卖出"** 与 **决策理由**。

- **数据全部真实 as-of**：`code/forecast.py` 打包决策日 10:00 PT 前可见信息——近 7 日 DA/RTPD/价差、同小时历史价差分布、目标日日前负荷预报（`load_2DA`，截至 2026-08-07）、目标日分区天气（截至 2026-08-19）；缺失部分如实进入 `coverage_warnings`（如 8/8~8/9 无 D-1 价格、无负荷预报），**绝不伪造**。
- **大模型推理**：`code/llm_forecast.py` 把数据包交给 LLM（DeepSeek，复用 `.env` 配置），输出结构化预测 JSON：`decision`（BUY_DA / SELL_DA / NO_TRADE）、预测 DA / RTPD / 价差、**建议买入价 / 卖出价**、把握度、2~5 条基于真实数字的理由与风险提示。
- **程序化校验（LLM 不可改判）**：决策枚举、价差与 DA−RTPD 自洽、预测价超出近 7 日均值 ± 4σ 时给出合理性警告（不否决但如实标注）、数据覆盖不足时不允许"高"把握度。
- **无 LLM 时诚实降级**：naive 基线（近 7 日均值）+ NO_TRADE，明确标注"LLM NOT CONFIGURED"。
- **诚实边界**：本功能是**独立的实验性推理层**，不触碰冻结的交易核心（模型 / Rule Engine / Risk Gate 不参与）；页面标注"不保证盈利"。数据现实：价格实际截至 2026-08-05，故 8/6~8/9 是真正的"未来"预测，无法用真实结果事后验证。

**LLM 环境变量（放在项目根 `.env`，已 gitignore）**：

```bash
LLM_PROVIDER=deepseek            # openai | deepseek | anthropic | mock
LLM_API_KEY=sk-xxxx              # API Key（必填，否则降级）
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
```

> 测试入口 `run_tests.py` 会强制测试进程无外部 LLM Key（保持离线确定性）；正常启动服务器时 `.env` 照常生效。

## 6. Data Sources / How It Works

- **Data Sources**：`/data-sources`（页面）逐字段登记来源（价格/负荷/天气/GFS 证据/节点映射），诚实标注"核心数据来自公司文件（已与 CAISO OASIS 对账），GFS 预报仅作 Agent Evidence、不进入模型特征"。
- **How It Works**：`/how-it-works`（页面）展示决策流水线、交易语义、Rule Engine 规则表（R-A~R-I）与诚实边界。
- 实现：`mvp_web.py`（Flask Web）→ `code/decision_service.py`（DecisionService + 6 Tool）→ `code/llm_copilot.py`（LLM Copilot）→ `agent/evidence/*`（统一 GFS Collector + Evidence Time Gate）→ `code/risk_gate/` + `code/decision/rule_engine.py`（裁决）。

## 7. MVP 简化（诚实说明）

1. **模型信号弱**：`CURRENT ALPHA = WEAK`，幅度/方向区分度都很有限；**不要拿它当赚钱信号**。
2. **真实外部证据有限**：唯一接入的真实源是 GFS 天气预报（Open-Meteo 历史档案，06Z 可严格回测），只报天气、不直接说价差方向；其余事件类源未接入。
3. **天气特征是"历史滞后"**：模型用 T−2 实际天气（`t2m_lag1` 等），不是决策时点能拿到的 D+1 预报档案。
4. **Risk Gate 是护栏不是发动机**：能把系统性亏损交易拦掉，但不会让弱信号变强；test 窗口 BUY 被系统性拒绝，保守是设计而非遗漏。
5. **PnL 简化**：1 MWh/仓，仅覆盖 DAM/RTPD 价差，不含 FMM/阻塞/费用等完整结算。
6. **Case Library 来自 test 窗口**：窗口早期几乎不触发，越往后越多。
7. **ELCA 冷启动**：样本不足，风控基本全拒。

## 8. 测试与审计

全仓库唯一测试入口（一次 discovery，无重复计数，TOTAL/PASSED/FAILED/SKIPPED 动态推导）：

```bash
python run_tests.py          # 全仓库 test_*.py 一次运行，唯一计数（当前 300 项全过）
python prepare_mvp.py        # 启动前自检
```

分块测试（按需）：
- V0.3.1.2 封板补丁 15 项验收：`python code/tests/test_v0312_freeze.py`
- V0.3.1.1 Demo Hardening 15 项验收：`python code/tests/test_v0311_hardening.py`
- V0.3.1 Web + LLM MVP 12 项验收：`python code/tests/test_mvp_v031.py`
- 回归：`python -m unittest code.tests.test_decision_service code.tests.test_llm_copilot`

覆盖要点：跨平台 canonical 哈希（CRLF/LF）、单一 DecisionService（CLI/Web/Tool 同一 DecisionSnapshot、无第二套 RiskGate/RuleEngine）、date-aware 市场规则版本（2026-05-01 边界）、available_at-only 证据判定、交易核心冻结（Golden 5 案例不变）、MOCK 隔离、Case 防穿越、LLM 不可覆盖工具数字。

## 9. 旧版指引

- CLI Demo（决策链单一来源 `DecisionService.run_decision` → DecisionSnapshot，与 Web/LLM Tools 同一份；本脚本仅渲染 + Lock/Reveal）：`python mvp_demo.py --help`；非技术用户指南见 `mvp_readme.md`。
- V0.2 技术文档：`docs/Architecture.md`、`docs/DecisionPipeline.md`、`docs/business_contract.md`、`docs/feature_availability_matrix.md` 等；`工程报告.md`、`数据审计与业务口径.md`。
