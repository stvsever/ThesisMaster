from __future__ import annotations


def test_pipeline_cli_accepts_frontend_root_overrides(module_loader, repo_root):
    module = module_loader(
        str(repo_root / "evaluation/00_pipeline_orchestration/run_pseudodata_to_impact.py"),
        "pipeline_cli_override_test_module",
    )

    args = module.parse_args(
        [
            "--initial-model-runs-root",
            "/tmp/custom_model_runs",
            "--free-text-root",
            "/tmp/custom_free_text",
        ]
    )
    assert args.initial_model_runs_root == "/tmp/custom_model_runs"
    assert args.free_text_root == "/tmp/custom_free_text"


def test_pipeline_cli_accepts_disable_llm_alias(module_loader, repo_root):
    module = module_loader(
        str(repo_root / "evaluation/00_pipeline_orchestration/run_pseudodata_to_impact.py"),
        "pipeline_cli_override_disable_llm_test_module",
    )
    args = module.parse_args(["--disable_LLM"])
    assert args.disable_llm is True


def test_launcher_cli_accepts_disable_llm_alias(module_loader, repo_root):
    module = module_loader(
        str(repo_root / "evaluation/00_pipeline_orchestration/run_pipeline.py"),
        "pipeline_launcher_disable_llm_alias_test_module",
    )
    args, passthrough = module.parse_args(["--disable_LLM", "--mode", "synthetic_v1"])
    assert args.disable_llm is True
    assert passthrough == []
