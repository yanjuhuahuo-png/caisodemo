# -*- coding: utf-8 -*-
"""
code/analysis/rolling_forecast_pnl.py —— 滚动重预测：实际价格 vs 预测价格（6 月 + 7 月）
=====================================================================================

滚动 walk-forward：对每个目标日 × 节点调用 LLM 预测层 forecast_day（每次重新打包
截至决策日的最新实际价格），得到该日预测 DA / RTPD / 价差；与当日实际 DA / RTPD /
价差对比，画在同一张折线图上（日期型横轴，覆盖用户指定区间）。

窗口：
  jun: 横轴 2026-06-01 ~ 07-01（目标日 06-02 ~ 07-01）
  jul: 横轴 2026-07-01 ~ 08-06（目标日 07-01 ~ 08-05，实际价格止于 08-05）

输出：code/analysis/pnl_charts_rolling/
  <node>_<win>_price.html     DA/RTPD 预测 vs 实际（4 条线，同图）
  <node>_<win>_spread.html    预测价差 vs 实际价差（2 条线，同图）
  summary_rolling_<win>.json  逐日明细 + 汇总
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from code.llm_forecast import forecast_day  # noqa: E402

NODES = ["CONTROLX_1_N001", "SNLNDRO_1_N001", "ELCAJNGT_7_N001"]
WINDOWS = [
    {"win": "jun", "target_start": date(2026, 6, 2), "target_end": date(2026, 7, 1),
     "axis": ["2026-06-01", "2026-07-01"]},
    {"win": "jul", "target_start": date(2026, 7, 1), "target_end": date(2026, 8, 5),
     "axis": ["2026-07-01", "2026-08-06"]},
]
OUT = Path(__file__).resolve().parent / "pnl_charts_rolling"
OUT.mkdir(exist_ok=True)

master = pd.read_csv(REPO / "code" / "data" / "master.csv", parse_dates=["date"])
master["date"] = pd.to_datetime(master["date"]).dt.date


def actual_daily(node: str, td: date) -> dict:
    sub = master[(master["node"] == node) & (master["date"] == td)]
    if sub.empty:
        return {}
    da = sub["da_price"].dropna()
    rt = sub["rtpd_price"].dropna()
    sp = (sub["da_price"] - sub["rtpd_price"]).dropna()
    return {
        "da_avg": round(float(da.mean()), 2) if len(da) else None,
        "rtpd_avg": round(float(rt.mean()), 2) if len(rt) else None,
        "spread_avg": round(float(sp.mean()), 2) if len(sp) else None,
    }


def main() -> None:
    for win in WINDOWS:
        summary = {}
        grand_actual = grand_pred = 0.0
        for node in NODES:
            rows = []
            td = win["target_start"]
            while td <= win["target_end"]:
                try:
                    r = forecast_day(td.isoformat(), node, trace=False)
                except Exception as exc:  # noqa: BLE001
                    rows.append({"target_date": str(td), "error": str(exc)})
                    td += timedelta(days=1)
                    continue
                f = r.get("forecast", {})
                a = actual_daily(node, td)
                rows.append({
                    "target_date": str(td),
                    "decision": r.get("decision"),
                    "da_pred": f.get("da_price_pred"), "rtpd_pred": f.get("rtpd_price_pred"),
                    "spread_pred": f.get("spread_pred"),
                    "da_actual": a.get("da_avg"), "rtpd_actual": a.get("rtpd_avg"),
                    "spread_actual": a.get("spread_avg"),
                })
                td += timedelta(days=1)

            df = pd.DataFrame([x for x in rows if "error" not in x])
            x = pd.to_datetime(df["target_date"])
            summary[node] = {
                "n": int(len(df)),
                "days": [dict(row) for row in df.to_dict("records")],
            }
            # ---- 图1：价格水平（DA / RTPD 预测 vs 实际，同一张图）----
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=df["da_actual"], name="实际 DA",
                                     mode="lines+markers", line=dict(color="#14845c", width=2.5)))
            fig.add_trace(go.Scatter(x=x, y=df["da_pred"], name="预测 DA（滚动）",
                                     mode="lines+markers", line=dict(color="#14845c", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=df["rtpd_actual"], name="实际 RTPD",
                                     mode="lines+markers", line=dict(color="#c0392b", width=2.5)))
            fig.add_trace(go.Scatter(x=x, y=df["rtpd_pred"], name="预测 RTPD（滚动）",
                                     mode="lines+markers", line=dict(color="#c0392b", width=2, dash="dash")))
            fig.update_xaxes(type="date", range=win["axis"], tickformat="%m-%d")
            fig.update_layout(
                title=f"{node} 滚动预测 vs 实际价格（{win['axis'][0]} ~ {win['axis'][1]}）",
                xaxis_title="目标日（横轴完整区间；边界无数据日留空）",
                yaxis_title="价格（$/MWh）",
                template="plotly_white", height=520, hovermode="x unified",
                legend=dict(orientation="h", y=1.12))
            fig.write_html(OUT / f"{node}_{win['win']}_price.html")

            # ---- 图2：价差（预测 vs 实际，同一张图）----
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=df["spread_actual"], name="实际价差",
                                      mode="lines+markers", line=dict(color="#14845c", width=2.5)))
            fig2.add_trace(go.Scatter(x=x, y=df["spread_pred"], name="预测价差（滚动）",
                                      mode="lines+markers", line=dict(color="#1f6feb", width=2.5, dash="dash")))
            fig2.add_hline(y=0, line_color="#9aa3af", line_width=1)
            fig2.update_xaxes(type="date", range=win["axis"], tickformat="%m-%d")
            fig2.update_layout(
                title=f"{node} 滚动预测 vs 实际价差（{win['axis'][0]} ~ {win['axis'][1]}）",
                template="plotly_white", height=460, hovermode="x unified",
                legend=dict(orientation="h", y=1.12))
            fig2.write_html(OUT / f"{node}_{win['win']}_spread.html")
            print(f"[{win['win']}] {node.replace('_1_N001','')}: {len(df)} 天 done")

        with open(OUT / f"summary_rolling_{win['win']}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"[{win['win']}] saved -> {OUT / ('summary_rolling_' + win['win'] + '.json')}")


if __name__ == "__main__":
    main()
