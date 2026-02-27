"""
Tests for the Refactoring Decision & Planning Agent (rdp_agent.py).
===================================================================

Run with:
    pytest test_rdp_agent.py -v
"""

import json
import os
import tempfile
from typing import Any, Dict

import pytest

from rdp_agent import (
    CodeSmell,
    QualityReport,
    RefactoringPlan,
    RefactoringStep,
    DEFAULT_CATALOG,
    DEFAULT_DEPENDENCIES,
    RATING_MAP,
    SEVERITY_ORDER,
    check_preconditions,
    generate_explanation,
    generate_plan,
    generate_plan_from_dict,
    score_candidate,
    select_best_candidate,
    sequence_steps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_smell_long_method() -> CodeSmell:
    """A Long Method smell with typical metrics."""
    return CodeSmell(
        id="smell_001",
        type="Long Method",
        location={"class": "Foo", "method": "bar", "lines": [10, 160]},
        metrics={"lines_of_code": 150, "cyclomatic_complexity": 30},
        severity="high",
    )


@pytest.fixture
def sample_smell_god_class() -> CodeSmell:
    """A God Class smell."""
    return CodeSmell(
        id="smell_002",
        type="God Class",
        location={"class": "BigClass", "method": None, "lines": [1, 800]},
        metrics={"lines_of_code": 800, "method_count": 20, "field_count": 15},
        severity="critical",
    )


@pytest.fixture
def sample_smell_feature_envy() -> CodeSmell:
    """A Feature Envy smell."""
    return CodeSmell(
        id="smell_003",
        type="Feature Envy",
        location={"class": "A", "method": "doStuff", "lines": [50, 80]},
        metrics={"external_field_accesses": 8, "lines_of_code": 30},
        severity="medium",
        details="TargetClass",
    )


@pytest.fixture
def sample_smell_duplicate() -> CodeSmell:
    return CodeSmell(
        id="smell_004",
        type="Duplicate Code",
        location={"class": "X", "method": "dup", "lines": [100, 130]},
        metrics={"lines_of_code": 30, "duplicate_blocks": 2},
        severity="medium",
    )


@pytest.fixture
def sample_smell_data_clumps() -> CodeSmell:
    return CodeSmell(
        id="smell_005",
        type="Data Clumps",
        location={"class": "Y", "method": "ship", "lines": [200, 220]},
        metrics={"parameter_count": 5, "lines_of_code": 20},
        severity="low",
    )


@pytest.fixture
def sample_smell_switch() -> CodeSmell:
    return CodeSmell(
        id="smell_006",
        type="Switch Statements",
        location={"class": "Z", "method": "pay", "lines": [300, 360]},
        metrics={"lines_of_code": 60, "cyclomatic_complexity": 10},
        severity="high",
    )


@pytest.fixture
def sample_quality_report_dict() -> Dict[str, Any]:
    """A minimal quality report dictionary with two smells."""
    return {
        "target": "Test.java",
        "smells": [
            {
                "id": "s1",
                "type": "Long Method",
                "location": {"class": "C", "method": "m", "lines": [1, 100]},
                "metrics": {"lines_of_code": 100, "cyclomatic_complexity": 15},
                "severity": "high",
            },
            {
                "id": "s2",
                "type": "Feature Envy",
                "location": {"class": "C", "method": "n", "lines": [200, 230]},
                "metrics": {"external_field_accesses": 5, "lines_of_code": 30},
                "severity": "medium",
                "details": "Other",
            },
        ],
        "metrics_summary": {"total_lines": 500},
    }


# ---------------------------------------------------------------------------
# Data Model Tests
# ---------------------------------------------------------------------------


class TestCodeSmell:
    """Tests for CodeSmell serialization."""

    def test_round_trip(self, sample_smell_long_method: CodeSmell) -> None:
        d = sample_smell_long_method.to_dict()
        restored = CodeSmell.from_dict(d)
        assert restored.id == sample_smell_long_method.id
        assert restored.type == sample_smell_long_method.type
        assert restored.severity == sample_smell_long_method.severity
        assert restored.metrics == sample_smell_long_method.metrics
        assert restored.location == sample_smell_long_method.location

    def test_optional_details_excluded(self, sample_smell_long_method: CodeSmell) -> None:
        d = sample_smell_long_method.to_dict()
        assert "details" not in d

    def test_details_included(self, sample_smell_feature_envy: CodeSmell) -> None:
        d = sample_smell_feature_envy.to_dict()
        assert d["details"] == "TargetClass"


class TestQualityReport:
    """Tests for QualityReport serialization."""

    def test_round_trip(self, sample_quality_report_dict: Dict[str, Any]) -> None:
        report = QualityReport.from_dict(sample_quality_report_dict)
        assert report.target == "Test.java"
        assert len(report.smells) == 2
        d = report.to_dict()
        assert d["target"] == "Test.java"
        assert len(d["smells"]) == 2


class TestRefactoringStep:
    """Tests for RefactoringStep serialization."""

    def test_round_trip(self) -> None:
        step = RefactoringStep(
            step_id=1,
            smell_id="s1",
            refactoring="Extract Method",
            target={"class": "C", "method": "m"},
            parameters={"source_lines": [10, 50]},
            explanation="Test explanation.",
        )
        d = step.to_dict()
        restored = RefactoringStep.from_dict(d)
        assert restored.step_id == 1
        assert restored.refactoring == "Extract Method"
        assert restored.parameters == {"source_lines": [10, 50]}


class TestRefactoringPlan:
    """Tests for RefactoringPlan serialization."""

    def test_round_trip(self) -> None:
        step = RefactoringStep(1, "s1", "Extract Method", {}, {}, "Expl.")
        plan = RefactoringPlan("plan_1", "File.java", [step], "Summary.")
        d = plan.to_dict()
        restored = RefactoringPlan.from_dict(d)
        assert restored.plan_id == "plan_1"
        assert len(restored.steps) == 1
        assert restored.summary == "Summary."


# ---------------------------------------------------------------------------
# Scoring Tests
# ---------------------------------------------------------------------------


class TestScoring:
    """Tests for the score_candidate function."""

    def test_low_complexity_low_risk_high_impact_is_best(self) -> None:
        candidate = {"complexity": "low", "risk": "low", "impact": "high"}
        score = score_candidate(candidate, CodeSmell("x", "Long Method", {}, {}, "high"))
        # (4-1)*0.2 + (4-1)*0.4 + 3*0.4 = 0.6 + 1.2 + 1.2 = 3.0
        assert abs(score - 3.0) < 1e-9

    def test_high_complexity_high_risk_low_impact_is_worst(self) -> None:
        candidate = {"complexity": "high", "risk": "high", "impact": "low"}
        score = score_candidate(candidate, CodeSmell("x", "Long Method", {}, {}, "high"))
        # (4-3)*0.2 + (4-3)*0.4 + 1*0.4 = 0.2 + 0.4 + 0.4 = 1.0
        assert abs(score - 1.0) < 1e-9

    def test_custom_weights(self) -> None:
        candidate = {"complexity": "low", "risk": "low", "impact": "high"}
        weights = {"complexity_weight": 0.5, "risk_weight": 0.3, "impact_weight": 0.2}
        smell = CodeSmell("x", "Long Method", {}, {}, "high")
        score = score_candidate(candidate, smell, weights)
        # (4-1)*0.5 + (4-1)*0.3 + 3*0.2 = 1.5 + 0.9 + 0.6 = 3.0
        assert abs(score - 3.0) < 1e-9

    def test_medium_defaults(self) -> None:
        candidate = {"complexity": "medium", "risk": "medium", "impact": "medium"}
        score = score_candidate(candidate, CodeSmell("x", "X", {}, {}, "low"))
        # (4-2)*0.2 + (4-2)*0.4 + 2*0.4 = 0.4 + 0.8 + 0.8 = 2.0
        assert abs(score - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Precondition Tests
# ---------------------------------------------------------------------------


class TestPreconditions:
    """Tests for precondition checking."""

    def test_has_code_block_true(self, sample_smell_long_method: CodeSmell) -> None:
        assert check_preconditions(["has_code_block"], sample_smell_long_method) is True

    def test_has_code_block_false_single_line(self) -> None:
        smell = CodeSmell("x", "Long Method", {"lines": [10, 11]}, {}, "low")
        assert check_preconditions(["has_code_block"], smell) is False

    def test_has_multiple_parameters_true(self, sample_smell_data_clumps: CodeSmell) -> None:
        assert check_preconditions(["has_multiple_parameters"], sample_smell_data_clumps) is True

    def test_has_multiple_parameters_false(self) -> None:
        smell = CodeSmell("x", "X", {}, {"parameter_count": 1}, "low")
        assert check_preconditions(["has_multiple_parameters"], smell) is False

    def test_has_multiple_responsibilities(self, sample_smell_god_class: CodeSmell) -> None:
        assert check_preconditions(["has_multiple_responsibilities"], sample_smell_god_class) is True

    def test_has_type_checking(self, sample_smell_switch: CodeSmell) -> None:
        assert check_preconditions(["has_type_checking"], sample_smell_switch) is True

    def test_empty_preconditions(self, sample_smell_long_method: CodeSmell) -> None:
        assert check_preconditions([], sample_smell_long_method) is True


# ---------------------------------------------------------------------------
# Candidate Selection Tests
# ---------------------------------------------------------------------------


class TestCandidateSelection:
    """Tests for select_best_candidate."""

    def test_long_method_selects_extract_method(
        self, sample_smell_long_method: CodeSmell
    ) -> None:
        best = select_best_candidate(sample_smell_long_method, DEFAULT_CATALOG)
        assert best is not None
        assert best["name"] == "Extract Method"

    def test_god_class_selects_extract_class(
        self, sample_smell_god_class: CodeSmell
    ) -> None:
        best = select_best_candidate(sample_smell_god_class, DEFAULT_CATALOG)
        assert best is not None
        assert best["name"] == "Extract Class"

    def test_feature_envy_selects_move_method(
        self, sample_smell_feature_envy: CodeSmell
    ) -> None:
        best = select_best_candidate(sample_smell_feature_envy, DEFAULT_CATALOG)
        assert best is not None
        assert best["name"] == "Move Method"

    def test_unknown_smell_returns_none(self) -> None:
        smell = CodeSmell("x", "UnknownSmell", {}, {}, "low")
        assert select_best_candidate(smell, DEFAULT_CATALOG) is None

    def test_switch_selects_polymorphism(self, sample_smell_switch: CodeSmell) -> None:
        best = select_best_candidate(sample_smell_switch, DEFAULT_CATALOG)
        assert best is not None
        assert best["name"] == "Replace Conditional with Polymorphism"


# ---------------------------------------------------------------------------
# Sequencing Tests
# ---------------------------------------------------------------------------


class TestSequencing:
    """Tests for dependency-aware step sequencing."""

    def test_extract_method_before_extract_class(self) -> None:
        smell_a = CodeSmell("a", "Long Method", {"class": "C", "method": "m", "lines": [1, 50]}, {"lines_of_code": 50}, "high")
        smell_b = CodeSmell("b", "God Class", {"class": "C", "lines": [1, 500]}, {"lines_of_code": 500, "method_count": 15}, "critical")

        cand_a = {"name": "Extract Method", "complexity": "low", "risk": "low", "impact": "high", "preconditions": []}
        cand_b = {"name": "Extract Class", "complexity": "high", "risk": "medium", "impact": "high", "preconditions": []}

        ordered = sequence_steps([(smell_a, cand_a), (smell_b, cand_b)])
        names = [c["name"] for _, c in ordered]
        assert names.index("Extract Method") < names.index("Extract Class")

    def test_single_item(self) -> None:
        smell = CodeSmell("a", "Long Method", {}, {}, "high")
        cand = {"name": "Extract Method", "complexity": "low", "risk": "low", "impact": "high", "preconditions": []}
        ordered = sequence_steps([(smell, cand)])
        assert len(ordered) == 1

    def test_empty_list(self) -> None:
        assert sequence_steps([]) == []

    def test_no_dependencies_ordered_by_severity(self) -> None:
        smell_low = CodeSmell("a", "X", {}, {}, "low")
        smell_high = CodeSmell("b", "Y", {}, {}, "high")
        cand_a = {"name": "Rename Method", "complexity": "low", "risk": "low", "impact": "low", "preconditions": []}
        cand_b = {"name": "Hide Delegate", "complexity": "low", "risk": "low", "impact": "medium", "preconditions": []}

        ordered = sequence_steps([(smell_low, cand_a), (smell_high, cand_b)])
        # No deps between them → higher severity first
        assert ordered[0][0].severity == "high"


# ---------------------------------------------------------------------------
# Explanation Tests
# ---------------------------------------------------------------------------


class TestExplanation:
    """Tests for explanation generation."""

    def test_contains_refactoring_name(
        self, sample_smell_long_method: CodeSmell
    ) -> None:
        cand = {"name": "Extract Method", "impact": "high", "risk": "low", "complexity": "low"}
        expl = generate_explanation(sample_smell_long_method, cand)
        assert "Extract Method" in expl

    def test_contains_smell_type(
        self, sample_smell_long_method: CodeSmell
    ) -> None:
        cand = {"name": "Extract Method", "impact": "high", "risk": "low", "complexity": "low"}
        expl = generate_explanation(sample_smell_long_method, cand)
        assert "Long Method" in expl

    def test_contains_metrics(self, sample_smell_long_method: CodeSmell) -> None:
        cand = {"name": "Extract Method", "impact": "high", "risk": "low", "complexity": "low"}
        expl = generate_explanation(sample_smell_long_method, cand)
        assert "150 lines of code" in expl
        assert "cyclomatic complexity 30" in expl


# ---------------------------------------------------------------------------
# End-to-End Test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration test: full pipeline with temp files."""

    def test_generate_plan_produces_valid_json(
        self, sample_quality_report_dict: Dict[str, Any]
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "report.json")
            output_path = os.path.join(tmpdir, "plan.json")

            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(sample_quality_report_dict, f)

            plan = generate_plan(input_path, output_path)

            # Output file should exist and be valid JSON
            assert os.path.isfile(output_path)
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "plan_id" in data
            assert data["target"] == "Test.java"
            assert isinstance(data["steps"], list)
            assert len(data["steps"]) > 0
            assert "summary" in data

            # Steps should reference smell IDs from input
            input_ids = {s["id"] for s in sample_quality_report_dict["smells"]}
            for step in data["steps"]:
                assert step["smell_id"] in input_ids

    def test_empty_smells(self) -> None:
        report = {"target": "Empty.java", "smells": [], "metrics_summary": {}}
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "report.json")
            output_path = os.path.join(tmpdir, "plan.json")

            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(report, f)

            plan = generate_plan(input_path, output_path)
            assert len(plan.steps) == 0
            assert "No applicable" in plan.summary

    def test_unknown_smell_type_skipped(self) -> None:
        report = {
            "target": "X.java",
            "smells": [
                {
                    "id": "u1",
                    "type": "CompletelyMadeUp",
                    "location": {"class": "X"},
                    "metrics": {},
                    "severity": "low",
                }
            ],
            "metrics_summary": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "report.json")
            output_path = os.path.join(tmpdir, "plan.json")

            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(report, f)

            plan = generate_plan(input_path, output_path)
            assert len(plan.steps) == 0


# ---------------------------------------------------------------------------
# file_name Fallback Tests
# ---------------------------------------------------------------------------


class TestFileNameFallback:
    """Tests for QualityReport file_name fallback support."""

    def test_file_name_only(self) -> None:
        """Report with file_name but no target should use file_name as target."""
        data = {
            "file_name": "PaymentService.java",
            "smells": [
                {
                    "id": "s1",
                    "type": "Long Method",
                    "location": {"class": "P", "method": "pay", "lines": [1, 50]},
                    "metrics": {"lines_of_code": 50},
                    "severity": "high",
                }
            ],
            "metrics_summary": {"total_lines": 200},
        }
        report = QualityReport.from_dict(data)
        assert report.target == "PaymentService.java"
        assert report.file_name == "PaymentService.java"

    def test_target_takes_precedence(self) -> None:
        """When both target and file_name are present, target wins."""
        data = {
            "target": "Primary.java",
            "file_name": "secondary.java",
            "smells": [],
            "metrics_summary": {},
        }
        report = QualityReport.from_dict(data)
        assert report.target == "Primary.java"
        assert report.file_name == "secondary.java"

    def test_neither_field(self) -> None:
        """When neither target nor file_name is present, fallback to 'unknown'."""
        data = {"smells": [], "metrics_summary": {}}
        report = QualityReport.from_dict(data)
        assert report.target == "unknown"

    def test_to_dict_includes_file_name(self) -> None:
        """to_dict should include file_name when set."""
        report = QualityReport(
            target="A.java",
            smells=[],
            metrics_summary={},
            file_name="A.java",
        )
        d = report.to_dict()
        assert d["file_name"] == "A.java"

    def test_to_dict_excludes_file_name_when_none(self) -> None:
        """to_dict should omit file_name when it's None."""
        report = QualityReport(target="A.java", smells=[], metrics_summary={})
        d = report.to_dict()
        assert "file_name" not in d

    def test_generate_plan_from_dict_with_file_name(self) -> None:
        """End-to-end: generate_plan_from_dict works with file_name field."""
        data = {
            "file_name": "CartService.java",
            "smells": [
                {
                    "id": "s1",
                    "type": "Long Method",
                    "location": {"class": "Cart", "method": "checkout", "lines": [1, 100]},
                    "metrics": {"lines_of_code": 100, "cyclomatic_complexity": 15},
                    "severity": "high",
                }
            ],
            "metrics_summary": {"total_lines": 300},
        }
        plan = generate_plan_from_dict(data)
        assert plan["target"] == "CartService.java"
        assert len(plan["steps"]) > 0
        assert "plan_id" in plan
