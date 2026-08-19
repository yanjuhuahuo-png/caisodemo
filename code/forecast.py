# -*- coding: utf-8 -*-
"""
code/forecast.py —— 未来交易日 LLM 预测的 as-of 数据打包（V0.4.3）
======================================================================

背景：用户要求在不重新训练模型的前提下，用仓库现有数据 + 大模型推理，
预测 2026-08-06 ~ 2026-08-09 逐日的电价，并给出"买入价 / 卖出价 / 决策理由"。

本模块只做**确定性数据打包**（无 LLM、无模型）：
  * 读取真实原始数据（价格 master.csv、日前负荷预报 load_2DA、分区天气
    zone_weather_hourly.csv），为指定 (target_date, node) 组装"决策时点
    as-of 可见"的信息包；
  * 所有时间均以 decision_date = target_date - 1 的 10:00 PT（DAM bid
    cutoff）为界：晚于该时点才发布的信息（如 target 日实际价格）绝不入包；
  * 数据覆盖如实标注：价格实际截至 2026-08-05，2DA 负荷预报截至 2026-08-07，
    天气截至 2026-08-19 —— 缺失部分进入 coverage_warnings，绝不伪造。

下游：code/llm_forecast.py 把本包喂给 LLM 生成预测；Web /forecast 页面展示。

与交易核心的关系：本模块**不改变** DecisionService / Rule Engine / Risk Gate /
模型信号（冻结）。LLM 预测是**独立的实验性推理层**，页面明确标注。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data_acquisition.schemas import NODE_REGION  # noqa: E402

MASTER_CSV = REPO_ROOT / "code" / "data" / "master.csv"
LOAD_2DA_CSV = REPO_ROOT / "load_CA_ISO_TAC_2DA.csv"
WEATHER_CSV = REPO_ROOT / "zone_weather_hourly.csv"

HOUR_LABELS = ["H%d" % h for h in range(1, 25)]

#: 预测窗口（用户需求：8/6 ~ 8/9；含 8/5 以便自选）
FORECAST_WINDOW = [f"2026-08-0{d}" for d in range(5, 10)]

#: 预测页可自选的日期范围（日历下拉框）
#:  min = 2024-01-10（保证有 ≥9 天价格历史供 as-of 参考）
#:  max = 2026-08-19（分区天气数据末日；更晚无任何预报输入）
FORECAST_MIN_DATE = "2024-01-10"
FORECAST_MAX_DATE = "2026-08-19"

_master_cache: Optional[pd.DataFrame] = None


def _load_master() -> pd.DataFrame:
    global _master_cache
    if _master_cache is None:
        m = pd.read_csv(MASTER_CSV, parse_dates=["date"])
        m = m.drop_duplicates(subset=["node", "date", "hour"], keep="first")
        _master_cache = m
    return _master_cache


def _load_2da_raw() -> pd.DataFrame:
    df = pd.read_csv(LOAD_2DA_CSV)
    df["date"] = pd.to_datetime(df["Date"].astype(str), format="mixed").dt.date
    df = df.melt(id_vars="date", value_vars=HOUR_LABELS, var_name="hc", value_name="load")
    df["hour"] = df["hc"].str.extract(r"(\d+)").astype(int)
    return df[["date", "hour", "load"]].dropna(subset=["load"])


def _load_weather_raw() -> pd.DataFrame:
    df = pd.read_csv(WEATHER_CSV)
    vt = pd.to_datetime(df["valid_pt"])
    df = df.assign(date=vt.dt.date, hour=vt.dt.hour + 1)
    return df[["zone", "date", "hour", "t2m_c", "ssrd_wm2", "wind100"]]


def _market_rule_for(target_date: str) -> str:
    try:
        from code.market_rules import market_rule_version_for  # noqa: PLC0415
        return market_rule_version_for(target_date)
    except Exception:
        return "UNKNOWN"


def build_forecast_package(target_date: str, node: str) -> Dict[str, Any]:
    """为 (target_date, node) 组装 as-of 数据包（确定性、真实、如实标注覆盖）。"""
    td = pd.Timestamp(target_date).date()
    dd = td - pd.Timedelta(days=1)  # decision_date
    zone = NODE_REGION.get(node, "?")

    master = _load_master()
    node_m = master[master["node"] == node].copy()
    node_m["date"] = pd.to_datetime(node_m["date"]).dt.date

    # ---- 1) 近期价格（as-of 保守口径：截至 target_date-2，即决策日前一日）----
    # 决策日 target_date-1 当天的 RTPD 要到决策日深夜才结算完整，10:00 PT 截止时
    # 不可见；故近期窗口 = target-8 .. target-2（预测 31 日 → 用 23~29 日），
    # 与 canonical 的 lag1（target_date-2 锚定）口径一致。
    hist_recent = node_m[node_m["date"] <= (td - pd.Timedelta(days=2))]
    recent: List[Dict[str, Any]] = []
    if len(hist_recent):
        daily = (hist_recent.groupby("date", as_index=False)
                 .agg(da=("da_price", "mean"), rt=("rtpd_price", "mean"),
                      spread=("spread", "mean")))
        daily = daily.sort_values("date").tail(7)
        for _, r in daily.iterrows():
            recent.append({
                "date": str(r["date"]),
                "da_avg": _f(r["da"]), "rtpd_avg": _f(r["rt"]), "spread_avg": _f(r["spread"]),
            })
    da_hist = np.array([r["da_avg"] for r in recent if r["da_avg"] is not None], dtype=float)
    rt_hist = np.array([r["rtpd_avg"] for r in recent if r["rtpd_avg"] is not None], dtype=float)
    sp_hist = np.array([r["spread_avg"] for r in recent if r["spread_avg"] is not None], dtype=float)

    # ---- 2) 同 node×hour 历史价差分布（全历史，as-of 语义 = 决策时可见）----
    # 该分布用"截至决策日"的完整历史（其他预测依赖不变），不受近期窗口调整影响。
    hist = node_m[node_m["date"] <= dd]
    hour_stats: List[Dict[str, Any]] = []
    for h in range(1, 25):
        sub = hist[hist["hour"] == h]["spread"].dropna().values
        if len(sub):
            hour_stats.append({
                "hour": h,
                "n": int(len(sub)),
                "p10": _f(float(np.percentile(sub, 10))),
                "p50": _f(float(np.percentile(sub, 50))),
                "p90": _f(float(np.percentile(sub, 90))),
                "std": _f(float(np.std(sub))),
            })

    # ---- 3) target 日负荷预报（load_2DA；真实"未来已知"信息，缺失如实标注）----
    l2 = _load_2da_raw()
    l2_t = l2[l2["date"] == td]
    load_forecast = None
    if len(l2_t):
        load_forecast = {
            "avg": _f(float(l2_t["load"].mean())),
            "peak_hour": int(l2_t.loc[l2_t["load"].idxmax(), "hour"]),
            "peak": _f(float(l2_t["load"].max())),
            "source": "load_CA_ISO_TAC_2DA.csv（官方日前负荷预测，决策时点可得）",
        }

    # ---- 4) target 日分区天气预报（zone_weather_hourly.csv）----
    w = _load_weather_raw()
    w_t = w[(w["zone"] == zone) & (w["date"] == td)]
    weather_forecast = None
    if len(w_t):
        weather_forecast = {
            "zone": zone,
            "t2m_c_avg": _f(float(w_t["t2m_c"].mean())),
            "t2m_c_max": _f(float(w_t["t2m_c"].max())),
            "ssrd_wm2_avg": _f(float(w_t["ssrd_wm2"].mean())),
            "wind100_avg": _f(float(w_t["wind100"].mean())),
            "source": "zone_weather_hourly.csv（逐小时天气）",
        }

    # ---- 5) 同 zone peer 节点近期价差（联动/阻塞信号）----
    peer = None
    peers = [n for n, z in NODE_REGION.items() if z == zone and n != node]
    if peers:
        pm = master[master["node"].isin(peers)].copy()
        pm["date"] = pd.to_datetime(pm["date"]).dt.date
        ph = pm[pm["date"] <= dd]
        if len(ph):
            pd_ = (ph.groupby("date", as_index=False).agg(spread=("spread", "mean"))
                   .sort_values("date").tail(3))
            peer = {
                "peers": peers,
                "recent_spread_avg": [{"date": str(r["date"]), "spread_avg": _f(r["spread"])}
                                      for _, r in pd_.iterrows()],
            }

    # ---- 6) 覆盖警告（诚实标注缺失）----
    warnings: List[str] = []
    last_price = str(node_m["date"].max()) if len(node_m) else "无"
    has_actual = len(node_m[node_m["date"] == td]) > 0
    if has_actual:
        warnings.append(f"target {target_date} 为历史日期：实际价格存在（事后可揭晓，但预测时不可见）。")
    else:
        warnings.append(f"价格数据实际截至 {last_price}；target {target_date} 的实际价格不存在（真实未来）。")
    last_price_d = node_m["date"].max() if len(node_m) else None
    lag1_d = td - pd.Timedelta(days=2)  # canonical lag1（保守对齐 target_date-2）
    if last_price_d is not None and lag1_d > last_price_d:
        warnings.append(
            f"决策时点可用的最近整日价格是 {last_price_d}，而滞后锚点 D-1={lag1_d} 在其后，"
            f"D-1 价格滞后特征缺失；LLM 需主要依赖负荷/天气预报与历史统计。")
    if load_forecast is None:
        warnings.append(f"target {target_date} 无日前负荷预报（load_2DA 截至 2026-08-07），负荷信号缺失。")
    if weather_forecast is None:
        warnings.append(f"target {target_date} 无分区天气（天气截至 2026-08-19）。")

    return {
        "target_date": str(td),
        "decision_date": str(dd),
        "decision_cutoff_pt": f"{dd} 10:00 PT（DAM bid cutoff）",
        "node": node,
        "zone": zone,
        "market_rule_version": _market_rule_for(str(td)),
        "as_of_banner": "AVAILABLE INFORMATION ONLY AS OF decision_date 10:00 PT",
        "recent_prices": recent,
        "recent_stats": {
            "da_mean": _f(float(np.mean(da_hist))) if len(da_hist) else None,
            "da_std": _f(float(np.std(da_hist))) if len(da_hist) else None,
            "rtpd_mean": _f(float(np.mean(rt_hist))) if len(rt_hist) else None,
            "rtpd_std": _f(float(np.std(rt_hist))) if len(rt_hist) else None,
            "spread_mean": _f(float(np.mean(sp_hist))) if len(sp_hist) else None,
            "spread_std": _f(float(np.std(sp_hist))) if len(sp_hist) else None,
        },
        # V0.4.5：衍生信号（帮 LLM 正确把握"该不该出手"）
        "volatility_class": _volatility_class(sp_hist),
        "recent_spread_trend": _recent_trend(recent),
        "same_hour_spread_stats": hour_stats,
        "load_forecast": load_forecast,
        "weather_forecast": weather_forecast,
        "peer_context": peer,
        "coverage_warnings": warnings,
        "honest_notes": [
            "LLM 推理预测为实验性功能，非系统冻结的交易核心，不保证盈利",
            "所有历史数字来自真实数据文件；缺失信息如实标注，绝不编造",
            "系统模型信号（ALPHA=WEAK）不参与此预测；此预测完全基于 LLM 推理 + 上述 as-of 数据",
        ],
    }


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if not np.isfinite(x):
            return None
        return round(x, 2)
    except (TypeError, ValueError):
        return None


def _volatility_class(sp_hist: np.ndarray) -> str:
    """近 7 日价差波动级别（低 <30 / 中 30~80 / 高 >80 $/MWh）。"""
    if len(sp_hist) < 3:
        return "低（样本不足）"
    std = float(np.std(sp_hist))
    if std > 80:
        return "高"
    if std >= 30:
        return "中"
    return "低"


def _recent_trend(recent: List[Dict[str, Any]]) -> Optional[str]:
    """近 3 个交易日的价差均值趋势（↑ / → / ↓ / 无数据）。"""
    vals = [r.get("spread_avg") for r in recent[-3:] if r.get("spread_avg") is not None]
    if len(vals) < 2:
        return None
    diff = vals[-1] - vals[0]
    if abs(diff) < 2.0:
        return f"持平（近3日价差均值 {vals[0]} → {vals[-1]}）"
    return f"近3日价差均值 {vals[0]} → {vals[-1]}（{'上行' if diff > 0 else '下行'} {abs(diff):.1f}）"


def fetch_actuals(target_date: str, node: str) -> Optional[Dict[str, Any]]:
    """目标日**事后实际结果**（仅当目标日真实存在于价格数据中才返回；未来日期 → None）。

    用于历史日期：预测后与真实结果对比（诚实标注"事后揭晓"，与决策无因果）。
    """
    td = pd.Timestamp(target_date).date()
    master = _load_master()
    node_m = master[master["node"] == node].copy()
    node_m["date"] = pd.to_datetime(node_m["date"]).dt.date
    sub = node_m[node_m["date"] == td]
    if not len(sub) or sub["da_price"].isna().all():
        return None
    hours = []
    for _, r in sub.sort_values("hour").iterrows():
        hours.append({"hour": int(r["hour"]), "da": _f(r["da_price"]),
                      "rtpd": _f(r["rtpd_price"]), "spread": _f(r["spread"])})
    da = [h["da"] for h in hours if h["da"] is not None]
    rt = [h["rtpd"] for h in hours if h["rtpd"] is not None]
    sp = [h["spread"] for h in hours if h["spread"] is not None]
    return {
        "target_date": str(td),
        "node": node,
        "note": "事后实际结果（价格数据截至 2026-08-05；仅历史日期可揭晓，与决策无因果）",
        "da_avg": _f(float(np.mean(da))) if da else None,
        "rtpd_avg": _f(float(np.mean(rt))) if rt else None,
        "spread_avg": _f(float(np.mean(sp))) if sp else None,
        "hours": hours,
    }


if __name__ == "__main__":
    import json
    for d in FORECAST_WINDOW:
        for n in ("CONTROLX_1_N001", "SNLNDRO_1_N001"):
            pkg = build_forecast_package(d, n)
            print(f"--- {d} {n} ---")
            print(json.dumps({k: pkg[k] for k in (
                "target_date", "decision_date", "recent_stats", "load_forecast",
                "weather_forecast", "coverage_warnings")}, ensure_ascii=False, indent=1)[:600])
