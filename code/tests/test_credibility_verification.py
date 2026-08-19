# -*- coding: utf-8 -*-
"""
code/tests/test_credibility_verification.py —— V0.4.2 结论可信度核验验收测试
============================================================================

新增能力：大模型 + 确定性门槛的"结论可信度核验"（verify_conclusion 工具 /
LLMCopilot.verify_credibility / Web /api/verify/<id> / Ask 预置路由）。

设计原则（与仓库"LLM 只解释不决策"一致）：
  * 可信度等级 = TRUSTWORTHY / CAUTION / NOT_TRUSTWORTHY，由**确定性数据完整性
    门槛**（audit 7 项运行时检查）计算；LLM 只能解释理由，**不可改判**。
  * 核验只依赖真实数据事实（audit / provenance / evidence Time Gate），
    LLM 输出必须与门槛一致，否则回退程序化理由。

覆盖（C1~C7）：
  C1  verify_conclusion 工具：全部审计 PASS → TRUSTWORTHY，事实字段齐备
  C2  verify_conclusion 工具：证据可用性 WARNING（真实历史快照 Case E）→ CAUTION
  C3  verify_credibility 降级：无 LLM → 程序化理由（llm_used=False）
  C4  verify_credibility 守卫：LLM 声称 TRUSTWORTHY 但门槛判定 NOT_TRUSTWORTHY → 不可改判
  C5  预置路由：含"可信 / 核验"关键词 → verify_conclusion（ask 走核验路径）
  C6  Web 端点 /api/verify/<id>：返回结构化核验结果
  C7  确定性理由与最终建议一致（conclusion 恒等于工具 final_recommendation）

运行（仓库根目录）：
    python code/tests/test_credibility_verification.py
    python -m unittest code.tests.test_credibility_verification -v

全部测试离线、确定性：决策用 StaticEvidenceAdapter / HistoricalSnapshotEvidenceAdapter
（本地文件，不联网）；LLM 一律 MockLlmClient 或显式 env={} 强制降级。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- 标准库 code 遮蔽防护（与 run_tests.py 同款）----
# 项目包名 `code` 与标准库同名；click.testing → pdb 导入时 `import code`。
# 独立运行本文件时，先移除仓库根路径让标准库 code 可解析，预导入 flask.testing
# （连带 pdb 完成导入），再恢复仓库根路径；项目包随后正常导入。
_guard_paths = list(sys.path)
sys.path = [p for p in sys.path if not (p and str(Path(p).resolve()) == str(REPO_ROOT))]
try:
    import code as _stdlib_code_guard  # noqa: F401
    import flask.testing  # noqa: F401
finally:
    sys.path = _guard_paths
    sys.modules.pop("code", None)      # 让项目包 code 可正常导入（pdb 已持有标准库引用）
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.decision_service import (  # noqa: E402
    DecisionService,
    HistoricalSnapshotEvidenceAdapter,
    StaticEvidenceAdapter,
)
from code.llm_copilot import (  # noqa: E402
    LLMCopilot,
    MockLlmClient,
    _match_preset,
)

DEMO_DIR = REPO_ROOT / "demo_artifacts"

DD = "2026-07-16"                 # 决策日（SELL 案例）
NODE = "CONTROLX_1_N001"
HOUR = 3
DD_E = "2026-07-08"               # 黄金案例 E（真实历史 GFS 18Z 被隔离）
HOUR_E = 2


def make_svc() -> DecisionService:
    return DecisionService(evidence_adapter=StaticEvidenceAdapter([]))


def make_demo_svc() -> DecisionService:
    return DecisionService(data_dir=DEMO_DIR,
                           evidence_adapter=HistoricalSnapshotEvidenceAdapter())


def run_did(svc, dd=DD, node=NODE, hour=HOUR) -> str:
    return svc.run_decision(dd, node, hour, reveal=False)["decision_id"]


class CredibilityVerificationTests(unittest.TestCase):
    """V0.4.2 结论可信度核验验收。"""

    @classmethod
    def setUpClass(cls):
        info = __import__("data_mode").resolve_data_mode()
        if not info.is_ready:
            raise unittest.SkipTest("数据不可用（mode=%s）：请先运行 prepare_mvp.py" % info.mode)

    # ============================================================= C1
    def test_c1_tool_trustworthy_when_audit_pass(self):
        """全部 7 项审计 PASS → TRUSTWORTHY；事实字段齐备且结论一致。"""
        svc = make_svc()
        did = run_did(svc)
        out = svc.verify_conclusion(did)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["verdict"]["level"], "TRUSTWORTHY")
        self.assertEqual(out["data_facts"]["audit_overall"], "PASS")
        self.assertEqual(out["data_facts"]["mock_used"], "NONE")
        self.assertEqual(out["data_facts"]["leakage_check"], "PASS")
        # 结论必须与最终建议一致（单一事实来源）
        self.assertEqual(out["conclusion"]["final_recommendation"],
                         svc._decisions[did]["final_recommendation"])  # noqa: SLF001
        self.assertTrue(out["data_facts"]["n_features"] > 0)
        self.assertIn("criteria", out["verdict"])
        self.assertTrue(out["verdict"]["criteria"])

    # ============================================================= C2
    def test_c2_tool_caution_when_evidence_availability_warning(self):
        """黄金案例 E：真实历史 GFS 18Z 快照被隔离且 available_at 未证明 → CAUTION。"""
        if not (DEMO_DIR / "evidence_demo.json").exists():
            self.skipTest("缺少 demo_artifacts/evidence_demo.json")
        svc = make_demo_svc()
        did = run_did(svc, dd=DD_E, hour=HOUR_E)
        out = svc.verify_conclusion(did)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["verdict"]["level"], "CAUTION")
        self.assertEqual(out["data_facts"]["audit_overall"], "WARNING")
        # 被隔离证据存在且未进决策
        self.assertGreaterEqual(out["data_facts"]["evidence_rejected"], 1)
        joined = " ".join(out["verdict"]["criteria"])
        self.assertIn("evidence_availability", joined)
        self.assertIn("WARNING", joined)

    # ============================================================= C3
    def test_c3_verify_degraded_uses_deterministic_reasons(self):
        """无 LLM → 程序化理由（llm_used=False），等级与工具门槛一致。"""
        svc = make_svc()
        did = run_did(svc)
        cp = LLMCopilot(service=svc, env={})  # env={} → 强制 NOT CONFIGURED
        r = cp.verify_credibility(decision_id=did, trace=True)
        self.assertEqual(r["status"], "ok")
        self.assertFalse(r["llm_used"])
        self.assertTrue(r["degraded"])
        tool = svc.verify_conclusion(did)
        self.assertEqual(r["verdict"]["level"], tool["verdict"]["level"])
        self.assertTrue(r["verdict"]["reasons"], "程序化理由不得为空")
        self.assertEqual(r["verdict"]["conclusion"],
                         tool["conclusion"]["final_recommendation"])
        self.assertTrue(r["trace"] and r["trace"]["steps"])

    # ============================================================= C4
    def test_c4_llm_cannot_override_not_trustworthy(self):
        """守卫：门槛判定 NOT_TRUSTWORTHY 时，LLM 声称 TRUSTWORTHY 不可改判。"""
        svc = make_svc()
        did = run_did(svc)
        # 篡改审计为 FAIL（模拟决策路径出现 MOCK / 泄漏等硬失败）
        dec = svc._decisions[did]  # noqa: SLF001
        dec["audit"]["overall"] = "FAIL"
        dec["audit"]["checks"]["mock_data"]["status"] = "FAIL"
        dec["audit"]["checks"]["mock_data"]["reason"] = "feature is_mock: fake"

        def responder(client, messages, tools):
            # LLM 恶意声称 TRUSTWORTHY（与确定性门槛冲突）
            return {"text": '{"verdict": "TRUSTWORTHY", "conclusion": "SELL_DA", '
                            '"reasons": ["数据看起来没问题"]}'}

        cp = LLMCopilot(service=svc, provider="mock",
                        llm_client=MockLlmClient(responder=responder), env={})
        r = cp.verify_credibility(decision_id=did, trace=True)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["verdict"]["level"], "NOT_TRUSTWORTHY",
                         "LLM 不可改判确定性门槛")
        self.assertFalse(r["llm_used"], "冲突输出应被守卫拦截，回退程序化理由")
        joined = "\n".join(r["verdict"]["reasons"])
        self.assertIn("不可采信", joined)
        self.assertIn("mock", joined.lower())

    # ============================================================= C5
    def test_c5_preset_route_matches_credibility_keywords(self):
        """含"可信 / 核验"关键词的问题路由到 verify_conclusion。"""
        for q in ("这个结论可信吗", "数据可信吗", "帮我核验一下结论", "结论靠谱吗"):
            with self.subTest(q=q):
                self.assertEqual(_match_preset(q), "verify_conclusion", q)
        # ask 走核验路径：工具包含 verify_conclusion
        svc = make_svc()
        cp = LLMCopilot(service=svc, env={})
        out = cp.ask("这个结论可信吗",
                     context={"decision_date": DD, "node": NODE, "hour": HOUR},
                     trace=True)
        tools = [t["tool"] for t in out["tools_called"]]
        self.assertIn("verify_conclusion", tools)
        self.assertIn("核验", out["answer"])

    # ============================================================= C6
    def test_c6_web_verify_endpoint(self):
        """/api/verify/<id> 返回结构化核验结果（无 LLM Key 时程序化）。"""
        import code.llm_copilot as _lc  # noqa: PLC0415
        with mock.patch.object(_lc, "_load_env_file", lambda: None), \
                mock.patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_PROVIDER": "",
                                             "LLM_MODEL": "", "LLM_BASE_URL": ""},
                                clear=False):
            _lc._default_copilot = None  # noqa: SLF001
            import mvp_web  # noqa: PLC0415
            c = mvp_web.app.test_client()
            did = c.post("/api/decision", json={"decision_date": DD, "node": NODE,
                                                "hour": HOUR, "evidence": "offline"}
                         ).get_json()["decision"]["decision_id"]
            r = c.post(f"/api/verify/{did}", json={})
            self.assertEqual(r.status_code, 200)
            j = r.get_json()
            self.assertEqual(j["status"], "ok")
            self.assertIn(j["verdict"]["level"], ("TRUSTWORTHY", "CAUTION", "NOT_TRUSTWORTHY"))
            self.assertTrue(j["verdict"]["reasons"])
            self.assertIn(j["verdict"]["conclusion"], ("BUY_DA", "SELL_DA", "NO_TRADE"))
            self.assertIn("audit_checks", j["facts"])
        # 未知 decision_id → 404
        r404 = c.post("/api/verify/does-not-exist", json={})
        self.assertEqual(r404.status_code, 404)

    # ============================================================= C7
    def test_c7_conclusion_consistency(self):
        """所有等级下，verdict.conclusion 恒等于最终建议（程序化单一事实来源）。"""
        for svc, dd, hour in ((make_svc(), DD, HOUR), (make_svc(), DD_E, HOUR_E)):
            did = run_did(svc, dd=dd, hour=hour)
            tool = svc.verify_conclusion(did)
            final = svc._decisions[did]["final_recommendation"]  # noqa: SLF001
            self.assertEqual(tool["conclusion"]["final_recommendation"], final)
            cp = LLMCopilot(service=svc, env={})
            r = cp.verify_credibility(decision_id=did)
            self.assertEqual(r["verdict"]["conclusion"], final)
            self.assertIn(r["verdict"]["level"], ("TRUSTWORTHY", "CAUTION", "NOT_TRUSTWORTHY"))


# ---------------------------------------------------------------------------
# 严格报告：TOTAL / PASSED / FAILED / SKIPPED
# ---------------------------------------------------------------------------
def main() -> int:
    import io  # noqa: PLC0415
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(CredibilityVerificationTests)
    runner = unittest.TextTestRunner(stream=io.StringIO(), verbosity=1)
    result = runner.run(suite)
    n_total = result.testsRun
    n_failed = len(result.failures) + len(result.errors)
    n_skipped = len(result.skipped)
    n_passed = n_total - n_failed - n_skipped
    print("=" * 64)
    print("V0.4.2 结论可信度核验验收（唯一计数）")
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
