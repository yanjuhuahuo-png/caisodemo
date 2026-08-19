"""一次性导出指定窗口的三节点组合每日盈亏表。"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

from code.decision_service import DecisionService, StaticEvidenceAdapter

start, end = date(2026, 6, 1), date(2026, 7, 1)
nodes = ["CONTROLX_1_N001", "SNLNDRO_1_N001", "ELCAJNGT_7_N001"]
service = DecisionService(evidence_adapter=StaticEvidenceAdapter([]))
rows = []
for offset in range((end - start).days + 1):
    target = start + timedelta(days=offset)
    actual = predicted = 0.0
    trades = missing = 0
    by_node = {}
    for node in nodes:
        node_actual = node_predicted = 0.0
        node_trades = 0
        for hour in range(1, 25):
            try:
                result = service.run_decision((target - timedelta(days=1)).isoformat(), node, hour, reveal=True)
            except ValueError:
                missing += 1
                continue
            pnl = (result.get("outcome") or {}).get("pnl")
            expected = result.get("prediction", {}).get("expected_return")
            action = result.get("final_recommendation")
            if pnl is not None:
                node_actual += float(pnl)
            if action == "SELL_DA" and expected is not None:
                node_predicted += float(expected)
                node_trades += 1
            elif action == "BUY_DA" and expected is not None:
                node_predicted -= float(expected)
                node_trades += 1
        actual += node_actual
        predicted += node_predicted
        trades += node_trades
        by_node[node] = {"actual_pnl": round(node_actual, 2), "pred_pnl": round(node_predicted, 2), "trades": node_trades}
    rows.append({"target_date": target.isoformat(), "actual_pnl": round(actual, 2),
                 "pred_pnl": round(predicted, 2), "trades": trades, "missing_hours": missing,
                 "by_node": by_node})
print(json.dumps(rows, ensure_ascii=False))
