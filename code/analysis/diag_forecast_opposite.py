# -*- coding: utf-8 -*-
"""
code/analysis/diag_forecast_opposite.py —— 诊断：LLM 预测与实际结果"完全相反"的原因
====================================================================================

对历史日期（≤2026-08-05，有事后实际）跑真实 LLM 预测，与逐日实际 DA/RTPD/价差对比：
  * 分类每笔：SAME（预测价差方向 == 实际方向）/ OPPOSITE（方向完全相反且实际幅度 ≥5）
              / NEUTRAL（实际幅度过小）
  * 对 OPPOSITE 案例提取归因特征：
      - 近期 7 日价差均值/标准差（波动性）
      - 趋势反转：近期 7 日均值方向 vs 实际方向
      - 数据覆盖警告数（缺失 D-1 价格 / 负荷 / 天气）
      - 实际幅度 vs 历史同小时 p90（是否尾事件）
  * 输出汇总 + 逐案例明细。

运行（真实 LLM，需 .env 配置）：python code/analysis/diag_forecast_opposite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from code.llm_forecast import forecast_day  # noqa: E402

NODES = ["CONTROLX_1_N001", "SNLNDRO_1_N001"]
DATES = [f"2026-07-{d:02d}" for d in range(28, 32)] + [f"2026-08-{d:02d}" for d in range(1, 6)]


def _sign(x):
    if x is None:
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def diagnose() -> dict:
    rows = []
    for td in DATES:
        for node in NODES:
            try:
                r = forecast_day(td, node, trace=False)
            except Exception as exc:  # noqa: BLE001
                rows.append({"target_date": td, "node": node, "error": str(exc)})
                continue
            a = r.get("actual")
            f = r.get("forecast", {})
            if not a or f.get("spread_pred") is None:
                rows.append({"target_date": td, "node": node,
                             "status": r["status"], "no_actual": a is None,
                             "decision": r.get("decision"), "skip": True})
                continue
            pkg = r.get("package", {})
            stats = pkg.get("recent_stats", {})
            f_sp = f["spread_pred"]
            a_sp = a.get("spread_avg")
            fs, as_ = _sign(f_sp), _sign(a_sp)
            if abs(a_sp or 0) < 5:
                cls = "NEUTRAL"
            elif fs != 0 and as_ != 0 and fs != as_:
                cls = "OPPOSITE"
            elif fs != 0 and as_ != 0 and fs == as_:
                cls = "SAME"
            else:
                cls = "NEUTRAL"
            # 归因特征
            r_mean = stats.get("spread_mean")
            r_std = stats.get("spread_std")
            trend_reversed = (r_mean is not None and _sign(r_mean) != 0
                              and as_ != 0 and _sign(r_mean) != as_)
            # 历史同小时 p90（目标小时取当日峰值小时近似：取所有小时 p90 的最大值）
            p90_max = max([h.get("p90") or 0 for h in pkg.get("same_hour_spread_stats", [])] or [0])
            extreme = abs(a_sp or 0) > p90_max
            rows.append({
                "target_date": td, "node": node, "status": r["status"],
                "decision": r.get("decision"), "class": cls,
                "f_spread": round(f_sp, 2), "a_spread": round(a_sp, 2),
                "f_da": f.get("da_price_pred"), "a_da": a.get("da_avg"),
                "recent_mean": r_mean, "recent_std": r_std,
                "trend_reversed": bool(trend_reversed),
                "n_warnings": len(pkg.get("coverage_warnings", [])),
                "extreme_actual": bool(extreme),
                "confidence": r.get("confidence"),
            })
    return rows


def main() -> None:
    rows = diagnose()
    classified = [r for r in rows if "class" in r]
    opp = [r for r in classified if r["class"] == "OPPOSITE"]
    same = [r for r in classified if r["class"] == "SAME"]
    neu = [r for r in classified if r["class"] == "NEUTRAL"]
    print("=" * 78)
    print("LLM 预测 vs 实际（历史窗口 %s ~ %s，%d 笔）" % (DATES[0], DATES[-1], len(classified)))
    print("=" * 78)
    print(f"SAME      : {len(same)} 笔（方向一致）")
    print(f"OPPOSITE  : {len(opp)} 笔（方向完全相反，实际|幅度|≥5）")
    print(f"NEUTRAL   : {len(neu)} 笔（实际幅度过小 / 无法判定）")
    print()
    if opp:
        print("--- OPPOSITE 案例明细与归因 ---")
        for r in opp:
            print(f"[{r['node'].replace('_1_N001','')}] {r['target_date']} "
                  f"决策={r['decision']} conf={r['confidence']} "
                  f"预测价差={r['f_spread']:+.1f} 实际价差={r['a_spread']:+.1f} "
                  f"(预测DA={r['f_da']} 实际DA={r['a_da']})")
            print(f"    近期7日均值={r['recent_mean']} std={r['recent_std']} "
                  f"| 趋势反转={r['trend_reversed']} | 覆盖警告={r['n_warnings']} | 实际为尾事件={r['extreme_actual']}")
        print()
        print("--- 归因统计（OPPOSITE vs SAME）---")
        for grp, label in ((opp, "OPPOSITE"), (same, "SAME")):
            stds = [r["recent_std"] for r in grp if r["recent_std"] is not None]
            rev = sum(1 for r in grp if r["trend_reversed"])
            warn = sum(1 for r in grp if r["n_warnings"] > 0)
            ext = sum(1 for r in grp if r["extreme_actual"])
            print(f"{label}: n={len(grp)} | 近7日std均值={sum(stds)/len(stds):.0f} "
                  f"| 趋势反转占比={rev}/{len(grp)} | 有覆盖警告={warn}/{len(grp)} | 实际尾事件={ext}/{len(grp)}")
    else:
        print("本窗口无 OPPOSITE 案例。")
    # 全部明细
    print()
    print("--- 全部明细 ---")
    for r in rows:
        if "class" in r:
            print(f"{r['node'].replace('_1_N001',''):>8} {r['target_date']} "
                  f"{r['class']:>9} | 预测={r['f_spread']:+.1f} 实际={r['a_spread']:+.1f} "
                  f"| 决策={r['decision']} conf={r['confidence']} | 7日均值={r['recent_mean']} "
                  f"std={r['recent_std']} 反转={r['trend_reversed']} 警告={r['n_warnings']}")


if __name__ == "__main__":
    main()
