"""
Tests for the pipeline_repo git history and deployment SHA cross-references.
"""

from __future__ import annotations
import subprocess
from pathlib import Path

import pytest

from generator.defaults import SIM_START, make_default_config
from generator.generate import generate_data
from generator.schema import query_deployments


REPO = str(Path(__file__).parent.parent / "pipeline_repo")
MODEL_DIR = str(Path(__file__).parent.parent / "model" / "artifacts")


def _git(args: list[str]) -> str:
    result = subprocess.run(["git"] + args, cwd=REPO,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


class TestPipelineRepoExists:
    def test_git_directory_present(self):
        assert (Path(REPO) / ".git").is_dir(), (
            "pipeline_repo/.git not found — run: python setup_pipeline_repo.py"
        )

    def test_at_least_seven_commits(self):
        log = _git(["log", "--oneline"])
        commits = [line for line in log.splitlines() if line.strip()]
        assert len(commits) >= 7, f"Expected ≥7 commits, got {len(commits)}"

    def test_feature_engineering_file_present(self):
        assert (Path(REPO) / "feature_engineering.py").is_file()

    def test_train_model_file_present(self):
        assert (Path(REPO) / "train_model.py").is_file()

    def test_serve_file_present(self):
        assert (Path(REPO) / "serve.py").is_file()


class TestDeploymentSHAIntegrity:
    @pytest.fixture(scope="class")
    def deploy_shas(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("deploy_sha"))
        config = make_default_config(n_days=1)
        config.output_dir = out
        db_paths = generate_data(config, MODEL_DIR, verbose=False)
        events = query_deployments(
            db_paths["deployments"],
            SIM_START, SIM_START + 7 * 86400,
        )
        return events

    def test_four_deploy_events(self, deploy_shas):
        assert len(deploy_shas) == 4

    def test_all_shas_are_valid_commits(self, deploy_shas):
        for event in deploy_shas:
            sha = event["commit_sha"]
            result = subprocess.run(
                ["git", "cat-file", "-t", sha],
                cwd=REPO, capture_output=True, text=True
            )
            assert result.returncode == 0, f"SHA {sha[:16]} not found in pipeline_repo"
            assert result.stdout.strip() == "commit", f"SHA {sha[:16]} is not a commit object"

    def test_change_types_are_correct(self, deploy_shas):
        change_types = {e["change_type"] for e in deploy_shas}
        expected = {"config_change", "feature_pipeline_change", "model_retrain", "dependency_bump"}
        assert change_types == expected

    def test_deploy_events_ordered_chronologically(self, deploy_shas):
        timestamps = [e["timestamp"] for e in deploy_shas]
        assert timestamps == sorted(timestamps)


class TestCodeDiffs:
    def _get_sha_before(self, sha: str) -> str:
        return _git(["rev-parse", f"{sha}^"])

    def test_feature_pipeline_diff_nonempty(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("diff_test"))
        config = make_default_config(n_days=1)
        config.output_dir = out
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        events = query_deployments(db_paths["deployments"], SIM_START, SIM_START + 7 * 86400)
        fp_event = next(e for e in events if e["change_type"] == "feature_pipeline_change")
        sha_after = fp_event["commit_sha"]
        sha_before = self._get_sha_before(sha_after)

        result = subprocess.run(
            ["git", "diff", sha_before, sha_after, "--", "feature_engineering.py"],
            cwd=REPO, capture_output=True, text=True
        )
        assert result.stdout.strip(), "Expected non-empty diff for feature_pipeline_change commit"

    def test_feature_pipeline_diff_adds_referral_source(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("diff_content"))
        config = make_default_config(n_days=1)
        config.output_dir = out
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        events = query_deployments(db_paths["deployments"], SIM_START, SIM_START + 7 * 86400)
        fp_event = next(e for e in events if e["change_type"] == "feature_pipeline_change")
        sha_after = fp_event["commit_sha"]
        sha_before = self._get_sha_before(sha_after)

        diff = subprocess.run(
            ["git", "diff", sha_before, sha_after, "--", "feature_engineering.py"],
            cwd=REPO, capture_output=True, text=True
        ).stdout
        assert '"referral_source"' in diff or "'referral_source'" in diff, (
            "Expected referral_source addition in feature_pipeline_change diff"
        )

    def test_dependency_bump_diff_shows_xgboost(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp("dep_diff"))
        config = make_default_config(n_days=1)
        config.output_dir = out
        db_paths = generate_data(config, MODEL_DIR, verbose=False)

        events = query_deployments(db_paths["deployments"], SIM_START, SIM_START + 7 * 86400)
        dep_event = next(e for e in events if e["change_type"] == "dependency_bump")
        sha_after = dep_event["commit_sha"]
        sha_before = self._get_sha_before(sha_after)

        diff = subprocess.run(
            ["git", "diff", sha_before, sha_after],
            cwd=REPO, capture_output=True, text=True
        ).stdout
        assert "xgboost" in diff, "Expected xgboost version change in dependency_bump diff"

    def test_diff_between_arbitrary_commits_works(self):
        log = _git(["log", "--oneline"])
        shas = [line.split()[0] for line in log.splitlines()]
        assert len(shas) >= 2
        diff = subprocess.run(
            ["git", "diff", shas[-1], shas[0]],
            cwd=REPO, capture_output=True, text=True
        )
        assert diff.returncode == 0
        assert diff.stdout.strip()  # some diff between first and last commit
