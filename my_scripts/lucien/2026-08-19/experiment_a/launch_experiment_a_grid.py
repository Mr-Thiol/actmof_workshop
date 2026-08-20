#!/usr/bin/env python3
"""Launch multiple Experiment-A settings in parallel.

Edit the CONFIGS and MAX_PARALLEL values below, then run this file from any
working directory:

    python my_scripts/lucien/2026-08-19/experiment_a/launch_experiment_a_grid.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# Directly edit these settings.
MAX_PARALLEL = 2
AUTO_PLOT = True
POLICY_NAMES = {"random", "gp_m52_pi", "gp_m52_ei", "imperfection_aware"}
CONFIGS = [
    {"policy": "gp_m52_pi", "n_rounds": 5, "budget": 150, "batch_size": 5, "exploration": 0},
    {"policy": "gp_m52_ei", "n_rounds": 5, "budget": 150, "batch_size": 5, "exploration": 0},
    {"policy": "imperfection_aware", "n_rounds": 5, "budget": 150, "batch_size": 5, "exploration": 1},
    {"policy": "random", "n_rounds": 5, "budget": 150, "batch_size": 5, "exploration": 0},
]


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
RUNNER = SCRIPT_DIR / "run_experiment_a.py"
PLOTTER = SCRIPT_DIR / "plot_experiment_a.py"
BASE_RESULTS_DIR = REPO_ROOT / "results" / "lucien" / "2026-08-20" / "experiment_a"
BASE_FIGURE_DIR = REPO_ROOT / "figures" / "lucien" / "2026-08-20" / "experiment_a"


@dataclass(frozen=True)
class ExperimentConfig:
    policy: str
    n_rounds: int
    budget: int
    batch_size: int
    exploration: int

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        return cls(
            policy=str(raw.get("policy", "imperfection_aware")),
            n_rounds=int(raw["n_rounds"]),
            budget=int(raw["budget"]),
            batch_size=int(raw["batch_size"]),
            exploration=int(raw["exploration"]),
        )

    @property
    def output_dir(self) -> Path:
        return (
            BASE_RESULTS_DIR
            / self.policy
            / f"n_rounds_{self.n_rounds}"
            / f"budget_{self.budget}"
            / f"batch_size_{self.batch_size}"
            / f"exploration_{self.exploration}"
        )

    @property
    def figure_dir(self) -> Path:
        return (
            BASE_FIGURE_DIR
            / self.policy
            / f"n_rounds_{self.n_rounds}"
            / f"budget_{self.budget}"
            / f"batch_size_{self.batch_size}"
            / f"exploration_{self.exploration}"
        )

    @property
    def label(self) -> str:
        return (
            f"policy={self.policy} n_rounds={self.n_rounds} budget={self.budget} "
            f"batch_size={self.batch_size} exploration={self.exploration}"
        )

    def validate(self) -> None:
        if self.policy not in POLICY_NAMES:
            raise ValueError(f"{self.label}: unknown policy")
        if self.n_rounds < 1:
            raise ValueError(f"{self.label}: n_rounds must be positive")
        if self.budget < 3:
            raise ValueError(f"{self.label}: budget must be at least 3")
        if self.batch_size < 1:
            raise ValueError(f"{self.label}: batch_size must be positive")
        if not 0 <= self.exploration <= self.batch_size:
            raise ValueError(f"{self.label}: exploration must be between 0 and batch_size")

    def command(self) -> list[str]:
        return [
            sys.executable,
            str(RUNNER),
            "--n-rounds",
            str(self.n_rounds),
            "--budget",
            str(self.budget),
            "--batch-size",
            str(self.batch_size),
            "--exploration",
            str(self.exploration),
            "--policy",
            self.policy,
            "--results-base-dir",
            str(BASE_RESULTS_DIR),
        ]

    def plot_command(self) -> list[str]:
        return [
            sys.executable,
            str(PLOTTER),
            "--results-dir",
            str(self.output_dir),
            "--figure-dir",
            str(self.figure_dir),
        ]

    @property
    def comparison_key(self) -> tuple[int, int, int]:
        return self.n_rounds, self.budget, self.batch_size

    @property
    def comparison_figure_dir(self) -> Path:
        return (
            BASE_FIGURE_DIR
            / "comparison"
            / f"n_rounds_{self.n_rounds}"
            / f"budget_{self.budget}"
            / f"batch_size_{self.batch_size}"
        )


@dataclass
class RunningJob:
    config: ExperimentConfig
    process: subprocess.Popen
    log_handle: object
    log_path: Path
    started_at: float


def launch(config: ExperimentConfig) -> RunningJob:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "launch.log"
    log_handle = log_path.open("w", encoding="utf-8")
    print(f"[START] {config.label}")
    print(f"[LOG]   {log_path}")
    log_handle.write(f"$ {' '.join(config.command())}\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        config.command(),
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return RunningJob(config, process, log_handle, log_path, time.time())


def finish(job: RunningJob) -> int:
    return_code = job.process.wait()
    job.log_handle.close()
    elapsed = time.time() - job.started_at
    status = "DONE" if return_code == 0 else "FAIL"
    print(f"[{status}] {job.config.label} rc={return_code} elapsed={elapsed / 60:.1f} min")
    print(f"[LOG]  {job.log_path}")
    if return_code == 0 and AUTO_PLOT:
        return_code = run_plot(job.config, job.log_path)
    return return_code


def run_plot(config: ExperimentConfig, log_path: Path) -> int:
    config.figure_dir.mkdir(parents=True, exist_ok=True)
    print(f"[PLOT] {config.label}")
    print(f"[FIG]  {config.figure_dir}")
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n$ {' '.join(config.plot_command())}\n\n")
        log_handle.flush()
        result = subprocess.run(
            config.plot_command(),
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        print(f"[PLOT-FAIL] {config.label} rc={result.returncode}")
    else:
        print(f"[PLOT-DONE] {config.label}")
    return result.returncode


def comparison_plot_command(config: ExperimentConfig, policies: list[str]) -> list[str]:
    return [
        sys.executable,
        str(PLOTTER),
        "--compare-policies",
        "--n-rounds",
        str(config.n_rounds),
        "--budget",
        str(config.budget),
        "--batch-size",
        str(config.batch_size),
        "--results-base-dir",
        str(BASE_RESULTS_DIR),
        "--figure-dir",
        str(config.comparison_figure_dir),
        "--policies",
        *policies,
    ]


def run_comparison_plots(configs: list[ExperimentConfig]) -> int:
    failures = 0
    grouped: dict[tuple[int, int, int], list[ExperimentConfig]] = {}
    for config in configs:
        grouped.setdefault(config.comparison_key, []).append(config)

    for group in grouped.values():
        first = group[0]
        policies = list(dict.fromkeys(config.policy for config in group))
        first.comparison_figure_dir.mkdir(parents=True, exist_ok=True)
        command = comparison_plot_command(first, policies)
        log_path = first.comparison_figure_dir / "comparison_plot.log"
        print(
            f"[COMPARE-PLOT] n_rounds={first.n_rounds} budget={first.budget} "
            f"batch_size={first.batch_size}"
        )
        print(f"[FIG]          {first.comparison_figure_dir}")
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"$ {' '.join(command)}\n\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode:
            failures += 1
            print(f"[COMPARE-PLOT-FAIL] rc={result.returncode}")
            print(f"[LOG]               {log_path}")
        else:
            print("[COMPARE-PLOT-DONE]")
    return failures


def stop_jobs(jobs: list[RunningJob]) -> None:
    for job in jobs:
        if job.process.poll() is None:
            job.process.terminate()
    for job in jobs:
        if job.process.poll() is None:
            try:
                job.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait()
        job.log_handle.close()
        print(f"[STOP] {job.config.label}")
        print(f"[LOG]  {job.log_path}")


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(__doc__)
        print("Edit CONFIGS and MAX_PARALLEL at the top of this file.")
        print("Set AUTO_PLOT to control automatic plotting after successful runs.")
        print("Use --dry-run to print commands without launching jobs.")
        return
    dry_run = "--dry-run" in sys.argv[1:]
    unknown_args = [arg for arg in sys.argv[1:] if arg != "--dry-run"]
    if unknown_args:
        raise SystemExit(f"Unknown launcher argument(s): {' '.join(unknown_args)}")
    if MAX_PARALLEL < 1:
        raise SystemExit("MAX_PARALLEL must be at least 1")

    configs = [ExperimentConfig.from_dict(raw) for raw in CONFIGS]
    for config in configs:
        config.validate()
    pending = list(configs)

    if dry_run:
        for config in pending:
            print(f"[DRY-RUN] {config.label}")
            print(f"[OUTPUT]  {config.output_dir}")
            print(f"[FIGURE]  {config.figure_dir}")
            print(f"[COMMAND] {' '.join(config.command())}")
            if AUTO_PLOT:
                print(f"[PLOT]    {' '.join(config.plot_command())}")
        if AUTO_PLOT:
            grouped: dict[tuple[int, int, int], list[ExperimentConfig]] = {}
            for config in pending:
                grouped.setdefault(config.comparison_key, []).append(config)
            for group in grouped.values():
                first = group[0]
                policies = list(dict.fromkeys(config.policy for config in group))
                command = comparison_plot_command(first, policies)
                print(f"[COMPARE] {' '.join(command)}")
        return

    running: list[RunningJob] = []
    failures = 0
    try:
        while pending or running:
            while pending and len(running) < MAX_PARALLEL:
                running.append(launch(pending.pop(0)))

            time.sleep(5)
            still_running = []
            for job in running:
                if job.process.poll() is None:
                    still_running.append(job)
                else:
                    failures += int(finish(job) != 0)
            running = still_running
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping running experiments")
        stop_jobs(running)
        raise SystemExit(130) from None

    if failures:
        raise SystemExit(f"{failures} experiment(s) failed")
    if AUTO_PLOT:
        failures = run_comparison_plots(configs)
        if failures:
            raise SystemExit(f"{failures} comparison plot(s) failed")
    print("[DONE] All experiments finished successfully")


if __name__ == "__main__":
    main()
