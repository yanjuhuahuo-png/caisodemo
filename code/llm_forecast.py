# -*- coding: utf-8 -*-
"""
code/llm_forecast.py —— LLM 驱动的未来交易日电价预测（V0.4.3，实验性推理层）
==============================================================================

需求：在不重新训练模型的前提下，用仓库现有数据 + 大模型思考能力，预测
2026-08-06 ~ 2026-08-09 逐日的电价，给出"多少钱买入 / 多少钱卖出"决策与理由。

职责：
  1. 调用 code/forecast.py 打包**真实 as-of 数据**（价格历史 / 负荷预报 /
     天气 / 历史统计 / 覆盖警告，全部确定性、无模型）；
  2. 把数据包交给 LLM（DeepSeek 等，复用 llm_copilot 的客户端与 .env 配置），
     要求其输出结构化预测 JSON：
       decision      : BUY_DA | SELL_DA | NO_TRADE
       da_price_pred : 预测日前均价（$/MWh）
       rtpd_price_pred: 预测实时均价（$/MWh）
       spread_pred   : 预测价差 = da - rtpd
       buy_price / sell_price : 决策方向下的建议买入 / 卖出价位
       confidence    : 低 | 中 | 高
       reasons       : 2~5 条，必须引用数据包中的真实数字
       caveats       : 风险提示
  3. **程序化校验与合理性检查**（LLM 不可改判的事实层）：
       - decision 必须是枚举值；价格必须为正有限数；
       - spread 与 da-rtpd 自洽；预测价与历史均值偏差超 4×std 时给出
         sanity_warning（不否决，但如实标注"超出历史范围"）；
       - 无 LLM（未配置）→ 诚实降级：用历史均值做 naive 基线 + NO_TRADE，
         明确标注"LLM NOT CONFIGURED，非推理预测"。

诚实边界（与仓库"LLM 只解释不决策"铁律的关系）：
  * 本模块是**独立的实验性推理层**，不触碰系统冻结的交易核心
    （DecisionService / Rule Engine / Risk Gate / 模型信号均不参与）；
  * 页面与返回结构一律标注 "LLM 推理预测 · 实验性 · 不保证盈利"；
  * 数据缺失如实标注（价格至 08-05 / 2DA 至 08-07 / 天气至 08-19）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.forecast import build_forecast_package, fetch_actuals  # noqa: E402

DECISIONS = ("BUY_DA", "SELL_DA", "NO_TRADE")
CONFIDENCES = ("低", "中", "高")

FORECAST_SYSTEM_PROMPT = (
    "You are a CAISO day-ahead electricity price forecaster. Your ONLY job: forecast "
    "tomorrow's (target day) electricity prices and a virtual trading decision, using "
    "STRICTLY the real as-of data package provided. You reason with your market knowledge; "
    "you never invent data facts that are not in the package.\n"
    "\n"
    "DECISION SEMANTICS (Convergence Bidding, 1 MWh):\n"
    "  SELL_DA: 预期 DA > RTPD → 卖出日前（虚拟供给），以 DA 价卖出、RTPD 价买回，赚 DA−RTPD。\n"
    "  BUY_DA : 预期 RTPD > DA → 买入日前（虚拟需求），以 DA 价买入、RTPD 价卖出，赚 RTPD−DA。\n"
    "  NO_TRADE: 判断不出 / 风险过大 / 数据不足 → 不交易（盈亏 0）。\n"
    "  buy_price / sell_price 含义（$/MWh，建议参考价位）：\n"
    "    SELL_DA → sell_price = 预测 DA 价（卖出日前），buy_price = 预测 RTPD 价（买回）。\n"
    "    BUY_DA  → buy_price = 预测 DA 价（买入日前），sell_price = 预测 RTPD 价（卖出）。\n"
    "    NO_TRADE → buy_price = sell_price = null。\n"
    "\n"
    "HARD RULES\n"
    "1. Output ONLY a JSON object, no markdown, no commentary:\n"
    '   {"decision":"BUY_DA|SELL_DA|NO_TRADE","da_price_pred":float,"rtpd_price_pred":float,'
    '"spread_pred":float,"buy_price":float|null,"sell_price":float|null,"confidence":"低|中|高",'
    '"reasons":["..."],"caveats":["..."]}\n'
    "2. All numbers must be derived from the data package (recent price levels, volatility, "
    "load forecast, weather, same-hour spread distribution). Do NOT state any fact not in the "
    "package (e.g. no invented news events, no invented historical values).\n"
    "3. spread_pred must be consistent with da_price_pred − rtpd_price_pred, and decision must "
    "match the spread sign: SELL_DA requires spread_pred > 0; BUY_DA requires spread_pred < 0.\n"
    "4. If the package has coverage_warnings (missing price lags / missing load), weigh them "
    "explicitly and lower confidence; if too little signal, output decision=NO_TRADE with reason.\n"
    "5. reasons: 2–5 concise Chinese sentences, each grounded in real numbers from the package. "
    "caveats: honest risks (data coverage gaps, experimental nature, ALPHA=WEAK).\n"
    "6. This is an experimental inference layer, NOT the system's frozen trading core; "
    "do not promise profit.\n"
    "7. VOLATILITY DISCIPLINE: if volatility_class is 高 (recent spread_std > 80 $/MWh), the "
    "spread direction is statistically near-unpredictable — default to decision=NO_TRADE unless "
    "load/weather/same-hour history gives an exceptionally strong reason, and never set "
    "confidence=高. When volatility_class is 中 or 高, prefer abstaining over a directional guess.\n"
    "8. TREND ANCHORING: start from recent_stats.spread_mean (and recent_spread_trend) as your "
    "base spread forecast; only deviate materially with a concrete reason (heat wave, load peak, "
    "negative-price pattern). Do NOT flip the recent trend sign on a whim.\n"
    "9. When uncertain about direction, NO_TRADE is the correct answer — a wrong directional "
    "bet is worse than no trade.\n"
)

NAIVE_FALLBACK = (
    "LLM NOT CONFIGURED —— 未配置 LLM（LLM_API_KEY / LLM_PROVIDER / LLM_MODEL），无法进行推理预测。\n"
    "以下为**确定性 naive 基线**（非推理）：以最近 7 日 DA/RTPD 均值作为预测价，"
    "因无判断依据，决策一律 NO_TRADE（不交易）。配置 LLM 后启用真实推理预测。"
)


# ---------------------------------------------------------------------------
# 程序化校验
# ---------------------------------------------------------------------------
def _parse_forecast_json(text: str) -> Optional[Dict[str, Any]]:
    t = str(text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(t[start:end + 1])
            except Exception:
                return None
        else:
            return None
    return obj if isinstance(obj, dict) else None


def _num(v: Any) -> Optional[float]:
    try:
        x = float(v)
        return x if x == x else None  # NaN → None
    except (TypeError, ValueError):
        return None


def validate_forecast(parsed: Dict[str, Any], pkg: Dict[str, Any]) -> tuple:
    """校验 LLM 预测与数据包一致性。返回 (ok, errors, sanity_warnings)。"""
    errors: List[str] = []
    sanity: List[str] = []

    decision = str(parsed.get("decision", "")).strip().upper()
    if decision not in DECISIONS:
        errors.append(f"decision 非法: {parsed.get('decision')!r}")
        return False, errors, sanity

    da = _num(parsed.get("da_price_pred"))
    rt = _num(parsed.get("rtpd_price_pred"))
    sp = _num(parsed.get("spread_pred"))
    # 注：CAISO 电价为负是真实现象（如 CONTROLX 近 7 日 DA 均值 -44.5），只校验有限数，
    # 不强制为正；极端值交由下方合理性带宽标注。
    if da is None:
        errors.append("da_price_pred 缺失/非数")
    if rt is None:
        errors.append("rtpd_price_pred 缺失/非数")
    if sp is None:
        errors.append("spread_pred 缺失/非数")
    if da is not None and rt is not None and sp is not None:
        if abs(sp - (da - rt)) > max(15.0, 0.5 * abs(da - rt)):
            errors.append(f"spread_pred({sp}) 与 da−rtpd({da - rt:.1f}) 不自洽")
    # 决策方向 ↔ 预测价差符号一致性（SELL 应 DA>RTPD；BUY 应 RTPD>DA）
    if sp is not None:
        if decision == "SELL_DA" and sp <= 0:
            errors.append(f"decision=SELL_DA 但 spread_pred={sp}<=0（卖出日前要求 DA>RTPD，自相矛盾）")
        if decision == "BUY_DA" and sp >= 0:
            errors.append(f"decision=BUY_DA 但 spread_pred={sp}>=0（买入日前要求 RTPD>DA，自相矛盾）")

    # 合理性：与最近 7 日历史均值的偏差超过 4×std → 标注（不否决）
    stats = pkg.get("recent_stats", {})
    for key, val, label in (("da_mean", da, "DA"), ("rtpd_mean", rt, "RTPD"), ("spread_mean", sp, "价差")):
        mean = _num(stats.get(key))
        std = _num(stats.get({  # noqa: C401 - 映射 std
            "da_mean": "da_std", "rtpd_mean": "rtpd_std", "spread_mean": "spread_std"}[key]))
        if val is not None and mean is not None and std is not None and std > 0:
            if abs(val - mean) > 4.0 * std:
                sanity.append(f"{label} 预测 {val:.1f} 超出近 7 日历史均值 {mean:.1f} ± 4σ({4 * std:.1f})，属极端情形，需谨慎")
    # 数据覆盖不足时不允许高置信度交易
    if pkg.get("coverage_warnings") and decision != "NO_TRADE":
        if str(parsed.get("confidence", "")).strip() == "高":
            sanity.append("数据包存在覆盖警告（缺失价格滞后/负荷），仍给出交易决策，置信度不应为'高'")
    # 高波动纪律（V0.4.5 程序化护栏）：波动级别"高"时方向统计上不可预测
    vc = str(pkg.get("volatility_class", "") or "")
    if decision != "NO_TRADE" and "高" in vc:
        sanity.append("近期价差波动为高（std>80 $/MWh），方向统计上不可预测，仍给出交易决策，失败风险高")
        if str(parsed.get("confidence", "")).strip() == "高":
            sanity.append("高波动场景下置信度不应为'高'")

    reasons = parsed.get("reasons")
    if not isinstance(reasons, list) or not reasons or not all(str(r).strip() for r in reasons):
        errors.append("reasons 必须为非空字符串列表")

    return (not errors), errors, sanity


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def forecast_day(target_date: str, node: str, trace: bool = True,
                 service=None, **kw) -> Dict[str, Any]:
    """预测 target_date（如 2026-08-06）节点 node 的电价与买卖决策。

    返回结构：
      status / package / decision / forecast(da/rtpd/spread) / prices(buy/sell) /
      confidence / reasons / caveats / sanity_warnings / errors / llm_used / degraded /
      trace（mode=forecast, steps）。
    """
    from code.llm_copilot import make_copilot  # noqa: PLC0415

    pkg = build_forecast_package(target_date, node)
    actual = fetch_actuals(target_date, node)   # 历史日期 → 事后实际；未来日期 → None
    trace_steps: List[Dict[str, Any]] = []
    cp = make_copilot(service=service, **kw)
    llm_used = False

    if cp.client is None or cp.degraded:
        # 诚实降级：naive 基线 + NO_TRADE
        stats = pkg.get("recent_stats", {})
        da = _num(stats.get("da_mean"))
        rt = _num(stats.get("rtpd_mean"))
        sp = _num(stats.get("spread_mean"))
        trace_steps.append({"stage": "llm", "status": "SKIPPED_DEGRADED",
                            "reason": "LLM NOT CONFIGURED（naive 基线降级）"})
        return {
            "status": "degraded",
            "answer": NAIVE_FALLBACK,
            "package": pkg,
            "actual": actual,
            "decision": "NO_TRADE",
            "forecast": {"da_price_pred": da, "rtpd_price_pred": rt, "spread_pred": sp},
            "prices": {"buy_price": None, "sell_price": None},
            "confidence": "低",
            "reasons": ["LLM 未配置，无推理依据；价格按最近 7 日均值 naive 估计，不构成交易依据。"],
            "caveats": ["naive 基线，非推理预测"],
            "sanity_warnings": [],
            "errors": [],
            "llm_used": False,
            "degraded": True,
            "trace": _trace_obj(trace, trace_steps, cp),
        }

    prompt = (
        "Forecast the target day and make a virtual trading decision. "
        "Use ONLY the real as-of data package below.\n\n"
        + json.dumps(pkg, ensure_ascii=False)
    )
    messages = [
        {"role": "system", "content": FORECAST_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        text = cp.client.chat(messages)
        trace_steps.append({"stage": "llm", "provider": cp.client.provider,
                            "model": cp.client.model, "mode": "forecast",
                            "final_answer_excerpt": str(text)[:300]})
    except Exception as exc:  # noqa: BLE001
        trace_steps.append({"stage": "llm", "status": "ERROR", "reason": str(exc)})
        return {
            "status": "llm_error",
            "answer": f"LLM ERROR: {type(exc).__name__}: {exc}",
            "package": pkg,
            "actual": actual,
            "decision": "NO_TRADE",
            "forecast": {}, "prices": {"buy_price": None, "sell_price": None},
            "confidence": "低", "reasons": ["LLM 调用失败，无法推理预测。"],
            "caveats": ["LLM 错误，未产生预测"],
            "sanity_warnings": [], "errors": [str(exc)],
            "llm_used": False, "degraded": False,
            "trace": _trace_obj(trace, trace_steps, cp),
        }

    parsed = _parse_forecast_json(text)
    if parsed is None:
        trace_steps.append({"stage": "guard", "status": "BLOCKED",
                            "reason": "LLM 输出非 JSON，无法解析"})
        return {
            "status": "guard_blocked",
            "answer": "UNCERTAIN / INSUFFICIENT EVIDENCE —— LLM 输出无法解析，未产生预测。",
            "package": pkg, "actual": actual, "decision": "NO_TRADE",
            "forecast": {}, "prices": {"buy_price": None, "sell_price": None},
            "confidence": "低", "reasons": ["LLM 输出格式非法，已拦截。"],
            "caveats": [], "sanity_warnings": [], "errors": ["非 JSON 输出"],
            "llm_used": False, "degraded": False,
            "trace": _trace_obj(trace, trace_steps, cp),
        }

    ok, errors, sanity = validate_forecast(parsed, pkg)
    if not ok:
        trace_steps.append({"stage": "guard", "status": "BLOCKED", "detail": "; ".join(errors)})
        return {
            "status": "guard_blocked",
            "answer": "UNCERTAIN / INSUFFICIENT EVIDENCE —— 预测未通过程序化校验（"
                     + "; ".join(errors) + "），已拦截。",
            "package": pkg, "actual": actual, "decision": "NO_TRADE",
            "forecast": {"da_price_pred": _num(parsed.get("da_price_pred")),
                         "rtpd_price_pred": _num(parsed.get("rtpd_price_pred")),
                         "spread_pred": _num(parsed.get("spread_pred"))},
            "prices": {"buy_price": _num(parsed.get("buy_price")),
                       "sell_price": _num(parsed.get("sell_price"))},
            "confidence": "低", "reasons": ["预测未通过程序化校验，已拦截。"],
            "caveats": [], "sanity_warnings": sanity, "errors": errors,
            "llm_used": True, "degraded": False,
            "trace": _trace_obj(trace, trace_steps, cp),
        }

    llm_used = True
    trace_steps.append({"stage": "guard", "status": "PASS"})
    return {
        "status": "ok",
        "answer": None,
        "package": pkg,
        "actual": actual,
        "decision": parsed["decision"],
        "forecast": {"da_price_pred": _num(parsed.get("da_price_pred")),
                     "rtpd_price_pred": _num(parsed.get("rtpd_price_pred")),
                     "spread_pred": _num(parsed.get("spread_pred"))},
        "prices": {"buy_price": _num(parsed.get("buy_price")),
                   "sell_price": _num(parsed.get("sell_price"))},
        "confidence": str(parsed.get("confidence", "中")),
        "reasons": [str(r) for r in parsed.get("reasons", [])],
        "caveats": [str(c) for c in parsed.get("caveats", [])],
        "sanity_warnings": sanity,
        "errors": [],
        "llm_used": True,
        "degraded": False,
        "trace": _trace_obj(trace, trace_steps, cp),
    }


def _trace_obj(trace_enabled: bool, steps: List[Dict[str, Any]], cp) -> Optional[Dict[str, Any]]:
    if not trace_enabled:
        return None
    numbered = []
    for i, s in enumerate(steps):
        s = dict(s)
        s["step"] = i + 1
        numbered.append(s)
    return {
        "user_question": "forecast_day（LLM 推理预测）",
        "mode": "forecast",
        "degraded": cp.degraded if hasattr(cp, "degraded") else False,
        "provider": cp.client.provider if cp.client else cp.provider,
        "model": cp.client.model if cp.client else cp.model,
        "steps": numbered,
    }


if __name__ == "__main__":
    import json as _json
    r = forecast_day("2026-08-06", "CONTROLX_1_N001")
    print(_json.dumps({k: r[k] for k in ("status", "decision", "forecast", "prices",
                                         "confidence", "reasons", "sanity_warnings",
                                         "llm_used", "degraded")},
                      ensure_ascii=False, indent=1, default=str))
