# -*- coding: utf-8 -*-
"""
code/analysis/pnl_curves.py —— 三节点"预测盈亏 vs 实际盈亏"曲线与汇总（2026-07 ~ 08-05）
=====================================================================================

用系统官方决策链（DecisionService.run_decision，FULL 数据，offline 证据）对
test 窗口（target 2026-07-01 ~ 2026-08-05）逐日逐小时出决策：
  - 实际盈亏 actual_pnl ：SELL_DA → +actual_return；BUY_DA → −actual_return；NO_TRADE → 0
  - 预测盈亏 pred_pnl   ：SELL_DA → +expected_return；BUY_DA → −expected_return；NO_TRADE → 0
按日汇总 → 三节点累计曲线（预测 vs 实际）+ 最终盈亏/交易统计。

输出：
  code/analysis/pnl_charts/<node>_pnl_curve.html  单节点每日：预测 vs 实际双曲线
  code/analysis/pnl_charts/all_nodes_pnl.html     三节点组合每日：预测 vs 实际双曲线
  code/analysis/pnl_summary.json                  汇总（最终盈亏/命中率/交易数）
运行：python code/analysis/pnl_curves.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from code.decision_service import DecisionService, StaticEvidenceAdapter  # noqa: E402

NODES = ["CONTROLX_1_N001", "SNLNDRO_1_N001", "ELCAJNGT_7_N001"]
START = date(2026, 7, 1)      # 首个 target_date
END = date(2026, 8, 5)        # 价格数据末日（实际盈亏仅到 08-05）
OUT_DIR = Path(__file__).resolve().parent / "pnl_charts"
OUT_DIR.mkdir(exist_ok=True)

COLORS = {"CONTROLX_1_N001": "#c0392b", "SNLNDRO_1_N001": "#14845c", "ELCAJNGT_7_N001": "#9a6b00"}


def _write_chart(path: Path, title: str, days, actual_label: str, pred_label: str) -> None:
    """写入不依赖第三方 JS 的 SVG 双折线图，确保前端可直接离线展示。"""
    import html

    width, height, left, right, top, bottom = 1120, 520, 70, 26, 80, 72
    values = [d["actual_pnl"] for d in days] + [d["pred_pnl"] for d in days] + [0]
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.12, 1)
    lo, hi = lo - pad, hi + pad
    plot_w, plot_h = width - left - right, height - top - bottom
    def x(i): return left + plot_w * i / max(len(days) - 1, 1)
    def y(v): return top + (hi - v) * plot_h / (hi - lo)
    def points(key): return " ".join(f"{x(i):.1f},{y(d[key]):.1f}" for i, d in enumerate(days))
    ticks = []
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        yy = y(v)
        ticks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>'
                     f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end">{v:+.1f}</text>')
    xlabels = []
    for i, d in enumerate(days):
        if i % 5 == 0 or i == len(days) - 1:
            xlabels.append(f'<text x="{x(i):.1f}" y="{height-40}" text-anchor="middle">{d["target_date"][5:]}</text>')
    rows = "".join(f'<tr><td>{d["target_date"]}</td><td>{d["actual_pnl"]:+.2f}</td><td>{d["pred_pnl"]:+.2f}</td><td>{d["trades"]}</td></tr>' for d in days)
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{margin:0;font-family:Arial,"Microsoft YaHei",sans-serif;color:#1f2937}}h2{{font-size:18px;margin:16px 20px 4px}}p{{margin:0 20px 10px;color:#667085;font-size:13px}}svg{{width:100%;height:auto}}text{{font-size:11px;fill:#667085}}.grid{{stroke:#e5e7eb;stroke-width:1}}.line-a{{fill:none;stroke:#14845c;stroke-width:3}}.line-p{{fill:none;stroke:#1f6feb;stroke-width:3;stroke-dasharray:8 5}}table{{border-collapse:collapse;margin:0 20px 18px;font-size:12px}}td,th{{padding:4px 10px;border:1px solid #e5e7eb;text-align:right}}th{{background:#f8fafc}}td:first-child{{text-align:left}}</style></head><body>
<h2>{html.escape(title)}</h2><p><span style="color:#14845c">● {actual_label}</span>&nbsp;&nbsp;<span style="color:#1f6feb">● {pred_label}</span>　单位：$/MWh，1 MWh/仓</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">{''.join(ticks)}<line x1="{left}" y1="{y(0):.1f}" x2="{width-right}" y2="{y(0):.1f}" stroke="#9aa3af"/><polyline points="{points('actual_pnl')}" class="line-a"/><polyline points="{points('pred_pnl')}" class="line-p"/>{''.join(xlabels)}<text x="18" y="{top}" transform="rotate(-90 18,{top})">当日盈亏</text></svg>
<table><thead><tr><th>目标日</th><th>实际盈亏</th><th>预测盈亏</th><th>交易小时</th></tr></thead><tbody>{rows}</tbody></table></body></html>'''
    path.write_text(document, encoding="utf-8")


def _chart(dates, days, node, total_actual, total_pred, traded_days, win_days, hit_rate):
    hit = f"{hit_rate:.0%}" if hit_rate is not None else "—"
    title = (f"{node} 每日盈亏：预测 vs 实际（2026-07-01 ~ 08-05）｜"
             f"实际最终 {total_actual:+,.2f} vs 预测最终 {total_pred:+,.2f} $/MWh｜交易 {traded_days} 天 · 命中率 {hit}")
    _write_chart(OUT_DIR / f"{node}_pnl_curve.html", title, days, "实际每日盈亏", "预测每日盈亏")


def _portfolio_chart(days, total_actual, total_pred, trade_days, trade_hours):
    title = ("三节点组合每日盈亏：预测 vs 实际（2026-07-01 ~ 08-05）｜"
             f"实际最终 {total_actual:+,.2f} vs 预测最终 {total_pred:+,.2f} $/MWh｜"
             f"交易 {trade_days} 天 / {trade_hours} 小时")
    _write_chart(OUT_DIR / "all_nodes_pnl.html", title, days, "实际每日盈亏（组合）", "预测每日盈亏（组合）")


def main() -> None:
    svc = DecisionService(evidence_adapter=StaticEvidenceAdapter([]))
    summary = {}
    portfolio_days = {}

    for node in NODES:
        days = []
        td = START
        while td <= END:
            dd = td - timedelta(days=1)
            day_actual = 0.0
            day_pred = 0.0
            trades = 0
            for h in range(1, 25):
                try:
                    dec = svc.run_decision(dd.isoformat(), node, h, reveal=True)
                except Exception:
                    continue
                final = dec.get("final_recommendation")
                er = dec.get("prediction", {}).get("expected_return")
                pnl = (dec.get("outcome") or {}).get("pnl")
                if pnl is None:
                    continue
                day_actual += float(pnl)
                if final == "SELL_DA" and er is not None:
                    day_pred += float(er); trades += 1
                elif final == "BUY_DA" and er is not None:
                    day_pred -= float(er); trades += 1
            days.append({
                "target_date": td.isoformat(),
                "actual_pnl": round(day_actual, 2),
                "pred_pnl": round(day_pred, 2),
                "trades": trades,
            })
            item = portfolio_days.setdefault(td.isoformat(), {
                "target_date": td.isoformat(), "actual_pnl": 0.0,
                "pred_pnl": 0.0, "trades": 0,
            })
            item["actual_pnl"] += day_actual
            item["pred_pnl"] += day_pred
            item["trades"] += trades
            td += timedelta(days=1)

        cum_actual = [0.0]
        cum_pred = [0.0]
        for d in days:
            cum_actual.append(cum_actual[-1] + d["actual_pnl"])
            cum_pred.append(cum_pred[-1] + d["pred_pnl"])
        dates = [d["target_date"][5:] for d in days]
        total_actual = cum_actual[-1]
        total_pred = cum_pred[-1]
        traded_days = [d for d in days if d["trades"] > 0]
        win_days = [d for d in traded_days if d["actual_pnl"] > 0]
        hit_rate = len(win_days) / len(traded_days) if traded_days else None
        summary[node] = {
            "total_actual_pnl": round(total_actual, 2),
            "total_pred_pnl": round(total_pred, 2),
            "traded_days": len(traded_days),
            "win_days": len(win_days),
            "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
            "note": ("风控全部拒绝，未交易" if not traded_days
                     else "正常交易（风控放行）"),
            "days": days,
        }
        _chart(dates, days, node, total_actual, total_pred,
               len(traded_days), len(win_days), hit_rate)
        print(f"[{node}] 实际累计 {total_actual:+,.2f} | 预测累计 {total_pred:+,.2f} | "
              f"交易 {len(traded_days)} 天 | 命中 {len(win_days)} 天"
              + (f" | 命中率 {hit_rate:.1%}" if hit_rate is not None else ""))

    combined_days = []
    for d in sorted(portfolio_days.values(), key=lambda x: x["target_date"]):
        d["actual_pnl"] = round(d["actual_pnl"], 2)
        d["pred_pnl"] = round(d["pred_pnl"], 2)
        combined_days.append(d)
    total_actual = round(sum(d["actual_pnl"] for d in combined_days), 2)
    total_pred = round(sum(d["pred_pnl"] for d in combined_days), 2)
    trade_days = sum(d["trades"] > 0 for d in combined_days)
    trade_hours = sum(d["trades"] for d in combined_days)
    summary["portfolio"] = {
        "total_actual_pnl": total_actual,
        "total_pred_pnl": total_pred,
        "traded_days": trade_days,
        "trade_hours": trade_hours,
        "note": "三节点组合；按每日、每小时决策的实际与预测盈亏合计",
        "days": combined_days,
    }
    _portfolio_chart(combined_days, total_actual, total_pred, trade_days, trade_hours)
    print(f"[portfolio] 实际最终 {total_actual:+,.2f} | 预测最终 {total_pred:+,.2f} | "
          f"交易 {trade_days} 天 / {trade_hours} 小时")

    with open(Path(__file__).resolve().parent / "pnl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("saved ->", OUT_DIR)
    print("summary ->", Path(__file__).resolve().parent / "pnl_summary.json")


if __name__ == "__main__":
    main()
