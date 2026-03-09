"""Continuous Trainer — iterative assess-select-generate-train loop.

Orchestrates multiple training rounds where the student is assessed after
each round, weak areas are identified, and training data is re-weighted
to target those weaknesses.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from torch.utils.data import DataLoader

from src.data.explanation_generator import ExplanationDataset, pregenerate_explanations
from src.training.adaptive_data_selector import AdaptiveDataSelector
from src.training.performance_tracker import PerformanceTracker
from src.training.student_assessor import StudentAssessor

logger = logging.getLogger(__name__)


@dataclass
class ContinuousConfig:
    max_rounds: int = 1000  # safety cap only — real stopping is convergence/target/budget
    target_accuracy: float = 0.8
    patience: int = 3
    min_delta: float = 0.01
    epochs_per_round: int = 1
    sft_warmup_epochs: int = 2  # SFT warmup before GRPO (0 to disable)
    benchmark_fraction: float = 0.2
    min_problems_per_round: int = 50
    max_problems_per_round: int = 500
    weak_threshold: float = 0.6
    regen_after_failures: int = 2
    min_budget_per_round: float = 1.0
    state_path: str = "continuous_state.json"
    metrics_path: str = "continuous_metrics.json"
    explanations_path: str = "continuous_explanations.jsonl"


class ContinuousTrainer:
    """Runs the iterative assess → select → generate → train loop."""

    def __init__(
        self,
        grpo_trainer,
        teacher,
        train_data: list,
        eval_data: list,
        cost_tracker,
        config: ContinuousConfig,
        collate_fn: Callable,
        resume: bool = True,
        curriculum_generator=None,
    ):
        self.grpo_trainer = grpo_trainer
        self.teacher = teacher
        self.train_data = train_data
        self.cost_tracker = cost_tracker
        self.config = config
        self.collate_fn = collate_fn
        self.curriculum_generator = curriculum_generator

        # Split eval data into benchmark (for assessment) and eval (for training eval)
        split = max(1, int(len(eval_data) * config.benchmark_fraction))
        self.benchmark_data = eval_data[:split]
        self.eval_data = eval_data[split:]

        # Sub-components
        self.assessor = StudentAssessor(
            student_model=grpo_trainer.student_model,
            benchmark_data=self.benchmark_data,
            weak_threshold=config.weak_threshold,
            regen_after_failures=config.regen_after_failures,
        )
        self.selector = AdaptiveDataSelector(
            train_data=train_data,
            min_problems=config.min_problems_per_round,
            max_problems=config.max_problems_per_round,
        )
        self.perf_tracker = PerformanceTracker(metrics_path=config.metrics_path)

        # Explanation dataset (load existing or start fresh)
        if os.path.exists(config.explanations_path):
            self.explanation_dataset = ExplanationDataset.load(config.explanations_path)
        else:
            self.explanation_dataset = ExplanationDataset()

        # State
        self.current_round = 0
        self.best_accuracy = 0.0

        if resume:
            self._load_state()

    def run(self) -> Dict:
        """Execute the continuous training loop. Returns summary dict.

        Training continues until one of these conditions is met:
        - Target accuracy reached
        - Budget exhausted
        - Performance converged (adaptive patience)
        - Safety cap hit (max_rounds — very high, should never be the real stop)
        """
        logger.info(
            "Starting continuous training: target=%.1f%%, patience=%d",
            self.config.target_accuracy * 100, self.config.patience,
        )
        logger.info("Benchmark: %d problems, Training pool: %d problems",
                     len(self.benchmark_data), len(self.train_data))

        stop_reason = "max_rounds"
        rounds_without_improvement = 0
        effective_patience = self.config.patience

        # SFT warmup on first round
        if self.current_round == 0 and self.config.sft_warmup_epochs > 0:
            logger.info("Running SFT warmup for %d epochs before GRPO...",
                        self.config.sft_warmup_epochs)
            warmup_loader = DataLoader(
                self.train_data,
                batch_size=self.grpo_trainer.config.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
            self.grpo_trainer.sft_warmup(warmup_loader, self.config.sft_warmup_epochs)

        try:
            for rnd in range(self.current_round, self.config.max_rounds):
                self.current_round = rnd
                round_start = time.time()

                # 1. Budget check
                if self.cost_tracker is not None:
                    summary = self.cost_tracker.get_summary()
                    remaining = summary.get('budget_remaining', float('inf'))
                    if remaining < self.config.min_budget_per_round:
                        logger.info("Budget exhausted (remaining=$%.2f). Stopping.", remaining)
                        stop_reason = "budget_exhausted"
                        break

                # 2. Pre-assessment
                logger.info("=== Round %d ===", rnd)
                pre_result = self.assessor.assess(rnd)
                pre_accuracy = pre_result.overall_accuracy

                # 3. Target check
                if pre_accuracy >= self.config.target_accuracy:
                    logger.info("Target accuracy %.1f%% reached! (%.1f%%)",
                                self.config.target_accuracy * 100, pre_accuracy * 100)
                    stop_reason = "target_reached"
                    self.perf_tracker.record_round(
                        round_number=rnd, pre_accuracy=pre_accuracy,
                        post_accuracy=pre_accuracy,
                        category_accuracies=pre_result.category_accuracies,
                    )
                    break

                # 4. Select data
                selection = self.selector.select(pre_result, self.explanation_dataset)

                # 4b. Generate targeted problems for weak areas
                if self.curriculum_generator and pre_result.weak_categories:
                    targeted_count = min(20, self.config.max_problems_per_round // 5)
                    new_problems = self.curriculum_generator.generate_targeted_problems(
                        weak_categories=pre_result.weak_categories,
                        count=targeted_count,
                    )
                    if new_problems:
                        logger.info("Generated %d targeted problems for weak areas",
                                    len(new_problems))
                        self.selector.add_data(new_problems)
                        selection.selected_data.extend(new_problems)

                # 5. Generate explanations for regen candidates
                cost_before = (self.cost_tracker.get_summary()['total_cost']
                               if self.cost_tracker else 0.0)
                self._generate_regen_explanations(selection.regen_candidates)
                self._generate_missing_explanations(selection.selected_data)

                # 6. Train
                self.grpo_trainer.explanation_dataset = self.explanation_dataset

                eval_loader = DataLoader(
                    self.eval_data,
                    batch_size=self.grpo_trainer.config.batch_size,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                ) if self.eval_data else None

                round_loader = DataLoader(
                    selection.selected_data,
                    batch_size=self.grpo_trainer.config.batch_size,
                    shuffle=True,
                    collate_fn=self.collate_fn,
                )

                self.grpo_trainer.train(
                    round_loader, eval_loader,
                    num_epochs=self.config.epochs_per_round,
                )

                # 7. Post-assessment
                post_result = self.assessor.assess(rnd)
                post_accuracy = post_result.overall_accuracy

                # 8. Record metrics
                cost_after = (self.cost_tracker.get_summary()['total_cost']
                              if self.cost_tracker else 0.0)
                self.perf_tracker.record_round(
                    round_number=rnd,
                    pre_accuracy=pre_accuracy,
                    post_accuracy=post_accuracy,
                    category_accuracies=post_result.category_accuracies,
                    training_stats={
                        'num_problems': len(selection.selected_data),
                        'regen_candidates': len(selection.regen_candidates),
                        'category_weights': selection.category_weights,
                        'duration_seconds': time.time() - round_start,
                    },
                    api_cost=cost_after - cost_before,
                )

                # 9. Adaptive patience: track improvement
                improved = post_accuracy > self.best_accuracy + self.config.min_delta
                marginal = post_accuracy > self.best_accuracy  # any gain, even tiny

                if improved:
                    rounds_without_improvement = 0
                    # Reset patience when making real progress
                    effective_patience = self.config.patience
                elif marginal:
                    # Still gaining slightly — extend patience
                    rounds_without_improvement = 0
                    effective_patience = min(
                        self.config.patience * 2,
                        effective_patience + 1,
                    )
                    logger.info("Marginal improvement — extending patience to %d",
                                effective_patience)
                else:
                    rounds_without_improvement += 1

                logger.info(
                    "Round %d complete: %.1f%% → %.1f%% (best=%.1f%%, "
                    "stall=%d/%d)",
                    rnd, pre_accuracy * 100, post_accuracy * 100,
                    max(self.best_accuracy, post_accuracy) * 100,
                    rounds_without_improvement, effective_patience,
                )

                # 10. Checkpoint best
                if post_accuracy > self.best_accuracy:
                    self.best_accuracy = post_accuracy
                    self.grpo_trainer.save_checkpoint(is_best=True)

                self._save_state()

                # 11. Convergence check with adaptive patience
                if rounds_without_improvement >= effective_patience:
                    logger.info(
                        "Converged — no meaningful improvement for %d rounds.",
                        effective_patience,
                    )
                    stop_reason = "converged"
                    break

        except KeyboardInterrupt:
            logger.info("Continuous training interrupted by user.")
            stop_reason = "interrupted"
        except Exception:
            self._save_state()
            raise

        summary = self.perf_tracker.generate_summary()
        summary['stop_reason'] = stop_reason
        logger.info("Continuous training finished: %s", summary)
        return summary

    def _generate_regen_explanations(self, candidates):
        """Generate fresh explanations for persistently failing problems."""
        if not candidates:
            return

        logger.info("Regenerating explanations for %d persistently failing problems", len(candidates))
        for candidate in candidates:
            dp = candidate.data_point
            for temp in candidate.temperatures:
                try:
                    explanation = self.teacher.generate_explanation(
                        question=dp.question,
                        answer=dp.solution,
                        temperature=temp,
                        use_cache=False,
                        subject=dp.subject,
                        difficulty=dp.difficulty,
                    )
                    if isinstance(explanation, dict):
                        explanation = explanation.get('explanation', str(explanation))

                    self.explanation_dataset.add(dp.question, dp.solution, explanation, temp)
                    self.explanation_dataset.save_incremental(
                        self.config.explanations_path,
                        {'question': dp.question, 'answer': dp.solution,
                         'explanation': explanation, 'temperature': temp},
                    )
                except Exception as e:
                    logger.warning("Regen failed for question: %s", e)

    def _generate_missing_explanations(self, selected_data):
        """Generate explanations for selected problems that have none yet."""
        missing = [dp for dp in selected_data
                   if not self.explanation_dataset.has_question(dp.question)]
        if not missing:
            return

        logger.info("Generating explanations for %d new problems", len(missing))
        self.explanation_dataset = pregenerate_explanations(
            teacher=self.teacher,
            data_points=missing,
            group_size=self.grpo_trainer.config.group_size,
            output_path=self.config.explanations_path,
        )
        # Reload full dataset (pregenerate appends to file)
        if os.path.exists(self.config.explanations_path):
            self.explanation_dataset = ExplanationDataset.load(self.config.explanations_path)

    def _save_state(self):
        """Save loop state for resume."""
        state = {
            'current_round': self.current_round + 1,  # next round to run
            'best_accuracy': self.best_accuracy,
            'assessor_state': self.assessor.to_dict(),
        }
        with open(self.config.state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Resume from saved state if available."""
        if not os.path.exists(self.config.state_path):
            return

        with open(self.config.state_path) as f:
            state = json.load(f)

        self.current_round = state.get('current_round', 0)
        self.best_accuracy = state.get('best_accuracy', 0.0)

        assessor_data = state.get('assessor_state')
        if assessor_data:
            self.assessor = StudentAssessor.from_dict(
                assessor_data, self.grpo_trainer.student_model, self.benchmark_data,
            )

        logger.info("Resumed from round %d (best=%.1f%%)",
                     self.current_round, self.best_accuracy * 100)
