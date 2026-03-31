# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    class _TorchStub:
        def __getattr__(self, _: str) -> Any:
            raise RuntimeError("torch is required for compare_perf torch e2e example")

    class _NNStub:
        class Module:
            pass

        def __getattr__(self, _: str) -> Any:
            raise RuntimeError("torch is required for compare_perf torch e2e example")

    torch = cast(Any, _TorchStub())
    nn = cast(Any, _NNStub())
    TORCH_AVAILABLE = False

from putils.compare_perf import dump_compare_perf_snapshot
from putils.compare_perf.align import align_modules
from putils.compare_perf.collector import TimingCollector, compare_perf
from putils.compare_perf.config import collect_runtime_metadata
from putils.compare_perf.diff import compute_diff
from putils.compare_perf.export import export_compare_result
from putils.compare_perf.hooks import model_forward_timing


REGRESSION_SCOPE_NAME = "encoder.layer.1.mlp"
AUTO_HOOK_MODULE_NAMES = {
    "encoder.layer.0.attn",
    REGRESSION_SCOPE_NAME,
    "head.proj",
}


def _match_module_or_descendant(module_name: str, *, roots: set[str]) -> bool:
    for root in roots:
        if module_name == root or module_name.startswith(f"{root}."):
            return True
    return False


class SleepInjectedSequential(nn.Sequential):
    def __init__(
        self,
        *modules: Any,
        scope_name: str,
        sleep_scope_name: str | None,
        sleep_seconds: float,
    ):
        super().__init__(*modules)
        self.scope_name = scope_name
        self.sleep_scope_name = sleep_scope_name
        self.sleep_seconds = float(sleep_seconds)

    def forward(self, input_value: Any) -> Any:
        if self.sleep_scope_name == self.scope_name and self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return super().forward(input_value)


class TinyTorchModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        sleep_scope_name: str | None,
        sleep_seconds: float,
    ):
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "layer": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "attn": nn.Linear(input_dim, hidden_dim),
                            }
                        ),
                        nn.ModuleDict(
                            {
                                "mlp": SleepInjectedSequential(
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.GELU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    scope_name=REGRESSION_SCOPE_NAME,
                                    sleep_scope_name=sleep_scope_name,
                                    sleep_seconds=sleep_seconds,
                                )
                            }
                        ),
                    ]
                )
            }
        )
        self.head = nn.ModuleDict(
            {
                "proj": nn.Linear(hidden_dim, output_dim),
            }
        )

    def forward(self, x: Any) -> Any:
        x = self.encoder["layer"][0]["attn"](x)
        x = self.encoder["layer"][1]["mlp"](x)
        x = self.head["proj"](x)

        return x


def _run_training(
    *,
    tag: str,
    output_dir: Path,
    model_state_dict: dict[str, Any],
    sleep_scope_name: str | None,
    sleep_seconds: float,
    steps: int,
    batch_size: int,
    input_dim: int,
    output_dim: int,
) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for compare_perf torch e2e example")

    torch.manual_seed(2026)

    collector = TimingCollector(sync_mode="none")
    model = TinyTorchModel(
        input_dim=input_dim,
        hidden_dim=32,
        output_dim=output_dim,
        sleep_scope_name=sleep_scope_name,
        sleep_seconds=sleep_seconds,
    )
    model.load_state_dict(model_state_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    with model_forward_timing(
        model,
        collector=collector,
        threshold_seconds=1e-9,
        leaf_only=False,
        module_filter=lambda module_name, _module: _match_module_or_descendant(
            module_name,
            roots=AUTO_HOOK_MODULE_NAMES,
        ),
    ):
        for step_idx in range(steps):
            batch_seed = 5000 + step_idx
            generator = torch.Generator().manual_seed(batch_seed)
            inputs = torch.randn(batch_size, input_dim, generator=generator)
            targets = torch.randn(batch_size, output_dim, generator=generator)

            with compare_perf("train.step", collector=collector, threshold_seconds=1e-9):
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)

                with compare_perf("train.loss", collector=collector, threshold_seconds=1e-9):
                    loss = criterion(outputs, targets)

                with compare_perf("train.backward", collector=collector, threshold_seconds=1e-9):
                    loss.backward()

                with compare_perf("train.optimizer.step", collector=collector, threshold_seconds=1e-9):
                    optimizer.step()

    run_metadata = collect_runtime_metadata()
    run_metadata["source"] = "compare_perf_torch_e2e_example"
    run_metadata["steps"] = int(steps)
    run_metadata["sleep_scope"] = sleep_scope_name
    run_metadata["sleep_seconds"] = float(sleep_seconds)

    snapshot_path = dump_compare_perf_snapshot(
        collector=collector,
        output_dir=output_dir,
        step=steps,
        tag=tag,
        filename_template="{tag}.json",
        run_metadata_extra=run_metadata,
    )
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def run_compare_perf_torch_e2e(
    *,
    output_dir: str,
    steps: int = 6,
    sleep_seconds: float = 0.03,
    top_n: int = 5,
) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is required for compare_perf torch e2e example")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "baseline.json"
    target_path = out_dir / "target.json"
    trace_path = out_dir / "trace.json"
    aligned_trace_path = out_dir / "trace_aligned.json"
    aligned_stack_trace_path = out_dir / "trace_aligned_stack.json"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"

    torch.manual_seed(123)
    init_model = TinyTorchModel(
        input_dim=16,
        hidden_dim=32,
        output_dim=8,
        sleep_scope_name=None,
        sleep_seconds=0.0,
    )
    initial_state = {
        name: param.detach().clone()
        for name, param in init_model.state_dict().items()
    }

    baseline_payload = _run_training(
        tag="baseline",
        output_dir=out_dir,
        model_state_dict=initial_state,
        sleep_scope_name=None,
        sleep_seconds=0.0,
        steps=steps,
        batch_size=8,
        input_dim=16,
        output_dim=8,
    )
    target_payload = _run_training(
        tag="target",
        output_dir=out_dir,
        model_state_dict=initial_state,
        sleep_scope_name=REGRESSION_SCOPE_NAME,
        sleep_seconds=sleep_seconds,
        steps=steps,
        batch_size=8,
        input_dim=16,
        output_dim=8,
    )

    baseline_summary = baseline_payload["summary"]
    target_summary = target_payload["summary"]
    alignment = align_modules(
        left_modules=[str(name) for name in baseline_summary.keys()],
        right_modules=[str(name) for name in target_summary.keys()],
    )
    diff_result = compute_diff(
        alignment=alignment,
        baseline_summary=baseline_summary,
        target_summary=target_summary,
    )

    timeline_events = [dict(item, pid=1, process_name="baseline") for item in baseline_payload["events"]]
    timeline_events.extend(dict(item, pid=2, process_name="target") for item in target_payload["events"])

    exported = export_compare_result(
        timeline_events=timeline_events,
        alignment=alignment,
        diff_result=diff_result,
        config_echo={
            "scenario": "torch_forward_backward_optimizer",
            "sleep_scope": REGRESSION_SCOPE_NAME,
            "sleep_seconds": sleep_seconds,
            "steps": steps,
            "baseline": str(baseline_path),
            "target": str(target_path),
        },
        trace_json_path=str(trace_path),
        aligned_trace_json_path=str(aligned_trace_path),
        aligned_stack_trace_json_path=str(aligned_stack_trace_path),
        summary_json_path=str(summary_json_path),
        summary_md_path=str(summary_md_path),
        top_n=top_n,
    )

    summary = exported["summary_json"]
    top_regressions = summary.get("top_regressions", [])

    print(f"baseline: {baseline_path}")
    print(f"target: {target_path}")
    print(f"trace: {trace_path}")
    print(f"trace_aligned: {aligned_trace_path}")
    print(f"trace_aligned_stack: {aligned_stack_trace_path}")
    print(f"summary_json: {summary_json_path}")
    print(f"summary_md: {summary_md_path}")
    print("top_regressions:")
    if top_regressions:
        for item in top_regressions:
            print(
                "  - "
                f"{item['baseline_module']} -> {item['target_module']}, "
                f"delta_ms={item['delta_ms']:.6f}, delta_pct={item['delta_pct']}"
            )
    else:
        print("  - (none)")

    return {
        "baseline": str(baseline_path),
        "target": str(target_path),
        "trace": str(trace_path),
        "trace_aligned": str(aligned_trace_path),
        "trace_aligned_stack": str(aligned_stack_trace_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "summary": summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run torch e2e compare_perf example")
    parser.add_argument("--output-dir", required=True, help="Directory for baseline/target and diff artifacts")
    parser.add_argument("--steps", type=int, default=6, help="Training steps for baseline and target")
    parser.add_argument("--sleep-seconds", type=float, default=0.03, help="Injected sleep in target regression scope")
    parser.add_argument("--top-n", type=int, default=5, help="Top N regressions/improvements in summary")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        run_compare_perf_torch_e2e(
            output_dir=args.output_dir,
            steps=args.steps,
            sleep_seconds=args.sleep_seconds,
            top_n=args.top_n,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
