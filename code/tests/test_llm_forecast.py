# -*- coding: utf-8 -*-
"""
code/tests/test_llm_forecast.py —— V0.4.3 未来交易日 LLM 推理预测验收测试
=======================================================================

覆盖（F1~F7）：
  F1  forecast.py 数据包确定性：8/6 含负荷预报+天气；8/9 负荷缺失且覆盖警告如实标注
  F2  forecast_day 降级（无 LLM）：naive 基线 + NO_TRADE（llm_used=False）
  F3  forecast_day + Mock LLM 合法 JSON → ok：决策/买卖价位/理由
  F4  forecast_day + Mock LLM 非法 JSON → guard_blocked（拦截）
  F5  forecast_day + Mock LLM 不自洽价差 → guard_blocked
  F6  Web 端点 /api/forecast-day 可达并返回结构化结果
  F7  Web 页面 /forecast 可达（含日期选择器）

全部离线、确定性：LLM 一律 MockLlmClient 或显式 env={}；不联网。
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- 标准库 code 遮蔽防护（与 run_tests.py 同款；独立运行本文件时也生效）----
_guard_paths = list(sys.path)
sys.path = [p for p in sys.path if not (p and str(Path(p).resolve()) == str(REPO_ROOT))]
try:
    import code as _stdlib_code_guard  # noqa: F401
    import flask.testing  # noqa: F401
finally:
    sys.path = _guard_paths
    sys.modules.pop("code", None)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.forecast import FORECAST_WINDOW, build_forecast_package  # noqa: E402
from code.llm_copilot import MockLlmClient  # noqa: E402
from code.llm_forecast import forecast_day, validate_forecast  # noqa: E402

NODE = "CONTROLX_1_N001"
T6, T9 = "2026-08-06", "2026-08-09"


def _ok_json(decision="SELL_DA", da=55.0, rt=10.0, spread=45.0):
    return json.dumps({
        "decision": decision, "da_price_pred": da, "rtpd_price_pred": rt,
        "spread_pred": spread, "buy_price": rt, "sell_price": da,
        "confidence": "中",
        "reasons": ["近 7 日 DA 均值 44.5、价差均值 43.8，正价差为主",
                    "8/6 天气炎热（t2m 25.9°C）推高日前需求",
                    "负荷预报日均 32698 MW 偏高，支撑价格"],
        "caveats": ["ALPHA=WEAK", "价格数据仅到 08-05"],
    }, ensure_ascii=False)


class LLMForecastTests(unittest.TestCase):
    """V0.4.3 未来交易日 LLM 推理预测验收。"""

    # ============================================================= F1
    def test_f1_package_deterministic_and_honest_coverage(self):
        p6 = build_forecast_package(T6, NODE)
        self.assertEqual(p6["target_date"], T6)
        self.assertEqual(p6["decision_date"], "2026-08-05")
        self.assertIsNotNone(p6["load_forecast"], "8/6 应有 2DA 负荷预报（至 08-07）")
        self.assertIsNotNone(p6["weather_forecast"])
        self.assertTrue(p6["recent_prices"], "应有近期价格历史")
        self.assertTrue(p6["same_hour_spread_stats"])

        p9 = build_forecast_package(T9, NODE)
        self.assertIsNone(p9["load_forecast"], "8/9 无负荷预报（2DA 至 08-07）→ 如实缺失")
        warns = "\n".join(p9["coverage_warnings"])
        self.assertIn("08-07", warns)          # 缺失来源如实标注
        self.assertTrue(any("实际价格不存在" in w for w in p9["coverage_warnings"]))
        # as-of：近期价格全部 ≤ 决策日（08-08）
        for d in p9["recent_prices"]:
            self.assertLessEqual(d["date"], "2026-08-05")
        # 历史日期（7/16）：警告应说"实际价格存在（事后可揭晓）"，不得误导为"不存在"
        ph = build_forecast_package("2026-07-16", NODE)
        hw = "\n".join(ph["coverage_warnings"])
        self.assertIn("历史日期", hw)
        self.assertNotIn("实际价格不存在", hw)

    # ============================================================= F2
    def test_f2_degraded_naive_fallback(self):
        r = forecast_day(T6, NODE, env={})
        self.assertEqual(r["status"], "degraded")
        self.assertEqual(r["decision"], "NO_TRADE")
        self.assertFalse(r["llm_used"])
        self.assertTrue(r["degraded"])
        self.assertIn("LLM NOT CONFIGURED", r["answer"])
        # naive 基线 = 最近 7 日均值
        pkg = build_forecast_package(T6, NODE)
        self.assertAlmostEqual(r["forecast"]["da_price_pred"],
                               pkg["recent_stats"]["da_mean"], delta=0.01)

    # ============================================================= F3
    def test_f3_mock_llm_ok(self):
        r = forecast_day(T6, NODE, provider="mock",
                         llm_client=MockLlmClient(responder=lambda c, m, t: {"text": _ok_json()}),
                         env={})
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["llm_used"])
        self.assertEqual(r["decision"], "SELL_DA")
        self.assertEqual(r["prices"]["sell_price"], 55.0)   # SELL: 卖出 DA 价
        self.assertEqual(r["prices"]["buy_price"], 10.0)    # SELL: 买回 RTPD 价
        self.assertGreaterEqual(len(r["reasons"]), 2)
        self.assertEqual(r["errors"], [])

    # ============================================================= F4
    def test_f4_mock_llm_bad_json_blocked(self):
        r = forecast_day(T6, NODE, provider="mock",
                         llm_client=MockLlmClient(responder=lambda c, m, t: {"text": "not json at all"}),
                         env={})
        self.assertEqual(r["status"], "guard_blocked")
        self.assertEqual(r["decision"], "NO_TRADE")

    # ============================================================= F5
    def test_f5_inconsistent_spread_blocked(self):
        # spread_pred=999 与 da−rtpd=45 严重不自洽 → 拦截
        r = forecast_day(T6, NODE, provider="mock",
                         llm_client=MockLlmClient(
                             responder=lambda c, m, t: {"text": _ok_json(spread=999.0)}),
                         env={})
        self.assertEqual(r["status"], "guard_blocked")
        self.assertTrue(any("不自洽" in e for e in r["errors"]))

    # ============================================================= F9
    def test_f9_decision_spread_contradiction_blocked(self):
        """决策方向与预测价差符号矛盾（SELL_DA 却 spread<0）→ 拦截（V0.4.4 修复）。"""
        bad = json.loads(_ok_json())
        bad["decision"] = "SELL_DA"
        bad["spread_pred"] = -35.0            # SELL 却负价差
        bad["da_price_pred"], bad["rtpd_price_pred"] = 50.0, 85.0  # DA<RTPD
        r = forecast_day(T6, NODE, provider="mock",
                         llm_client=MockLlmClient(
                             responder=lambda c, m, t: {"text": json.dumps(bad, ensure_ascii=False)}),
                         env={})
        self.assertEqual(r["status"], "guard_blocked")
        self.assertTrue(any("自相矛盾" in e for e in r["errors"]), r["errors"])

    # ============================================================= F6
    def test_f6_web_endpoint(self):
        import code.llm_copilot as _lc  # noqa: PLC0415
        with mock.patch.object(_lc, "_load_env_file", lambda: None), \
                mock.patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_PROVIDER": "",
                                             "LLM_MODEL": "", "LLM_BASE_URL": ""},
                                clear=False):
            _lc._default_copilot = None  # noqa: SLF001
            import mvp_web  # noqa: PLC0415
            c = mvp_web.app.test_client()
            r = c.post("/api/forecast-day", json={"target_date": T6, "node": NODE})
            self.assertEqual(r.status_code, 200)
            j = r.get_json()
            self.assertEqual(j["status"], "degraded")          # 无 LLM → 诚实降级
            self.assertTrue(j["degraded"])
            self.assertEqual(j["decision"], "NO_TRADE")
            self.assertIn("forecast", j)
            self.assertIn("package", j)
            self.assertEqual(j["package"]["target_date"], T6)
            # 非法节点 → 结构化错误
            r2 = c.post("/api/forecast-day", json={"target_date": T6, "node": "NOPE"})
            self.assertEqual(r2.status_code, 400)
            self.assertEqual(r2.get_json()["error"]["code"], "UNSUPPORTED_NODE")

    # ============================================================= F7
    def test_f7_forecast_page_is_main_index(self):
        """主页面 "/" = 预测页（日历下拉框自选日期）；旧工作台入口链接已删除（V0.4.4）。"""
        import mvp_web  # noqa: PLC0415
        c = mvp_web.app.test_client()
        r = c.get("/")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("CAISO 电价预测", html)
        self.assertIn('type="date"', html)        # 日历下拉框
        self.assertIn("inp-td", html)
        self.assertIn("api/forecast-day", html)
        self.assertNotIn("旧决策工作台", html)     # 旧工作台前端入口已删除
        self.assertNotIn("系统如何做决策", html)   # 该入口已删除（V0.4.4）
        self.assertIn("数据来源", html)            # 仅保留数据来源入口
        # /forecast 别名仍可用
        r2 = c.get("/forecast")
        self.assertEqual(r2.status_code, 200)
        self.assertIn("inp-td", r2.get_data(as_text=True))
        # 旧工作台后端路由仍保留（决策 API 演示用，无前端入口）
        r3 = c.get("/decision-workspace")
        self.assertEqual(r3.status_code, 200)
        self.assertIn("sys-grid", r3.get_data(as_text=True))

    # ============================================================= F8
    def test_f8_historical_date_has_actuals(self):
        """历史日期（≤2026-08-05）预测结果附带事后实际；未来日期 actual=None。"""
        # 历史日期：2026-07-16（SELL 黄金案例日）
        r = forecast_day("2026-07-16", NODE, env={})
        self.assertEqual(r["status"], "degraded")
        a = r["actual"]
        self.assertIsNotNone(a, "历史日期应返回事后实际")
        self.assertEqual(a["target_date"], "2026-07-16")
        self.assertEqual(a["node"], NODE)
        self.assertIsNotNone(a["da_avg"])
        self.assertTrue(a["hours"], "应有逐小时实际")
        # 未来日期：2026-08-06（价格数据截止之后）→ 无实际
        r2 = forecast_day("2026-08-06", NODE, env={})
        self.assertIsNone(r2["actual"])
        # Web 端点透传
        import code.llm_copilot as _lc  # noqa: PLC0415
        with mock.patch.object(_lc, "_load_env_file", lambda: None), \
                mock.patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_PROVIDER": "",
                                             "LLM_MODEL": "", "LLM_BASE_URL": ""},
                                clear=False):
            _lc._default_copilot = None  # noqa: SLF001
            import mvp_web  # noqa: PLC0415
            c = mvp_web.app.test_client()
            j = c.post("/api/forecast-day", json={"target_date": "2026-07-16",
                                                  "node": NODE}).get_json()
            self.assertIsNotNone(j.get("actual"))
            self.assertEqual(j["actual"]["target_date"], "2026-07-16")


# ---------------------------------------------------------------------------
# 严格报告：TOTAL / PASSED / FAILED / SKIPPED
# ---------------------------------------------------------------------------
def main() -> int:
    import io  # noqa: PLC0415
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(LLMForecastTests)
    runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=1)
    result = runner.run(suite)
    n_total = result.testsRun
    n_failed = len(result.failures) + len(result.errors)
    n_skipped = len(result.skipped)
    n_passed = n_total - n_failed - n_skipped
    print("=" * 64)
    print("V0.4.3 未来交易日 LLM 推理预测验收（唯一计数）")
    print(f"TOTAL   : {n_total}")
    print(f"PASSED  : {n_passed}")
    print(f"FAILED  : {n_failed}")
    print(f"SKIPPED : {n_skipped}")
    print("=" * 64)
    if result.failures or result.errors:
        for t, tb in result.failures + result.errors:
            print(f"  FAILED  {t.id()}")
            print("  " + "\n  ".join(tb.splitlines()[-8:]))
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
