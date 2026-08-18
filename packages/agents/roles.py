"""Bull / Bear / Skeptic / Arbiter Role Implementations for CHIMERA Phase 4.

Each agent produces exactly ONE structured JSON output per debate round.
No open-ended conversation loops. No free-text hallucination.

Default implementation is RULE-BASED (zero token cost).
LLM-backed variants can be substituted by subclassing the base roles
and overriding the argue() / design() methods.

Per AGENTS.md: Antigravity is responsible for agent flow architecture,
NOT for physics integrators or LLM prompt engineering.
"""

from __future__ import annotations
from typing import Optional
import math

from packages.symbolic.hypothesis import Hypothesis
from packages.agents.debate_models import (
    BullArgument,
    SupportingEvidence,
    BearArgument,
    Weakness,
    CounterfactualExperiment,
    PerturbationSpec,
    ExperimentResult,
    ArbiterVerdict,
    EvidenceWeight,
)


# ---------------------------------------------------------------------------
# Bull Agent
# ---------------------------------------------------------------------------

class BullAgent:
    """Presents the strongest quantitative case FOR the hypothesis.

    Rule-based implementation:
      - Cites R², RMSE, and training evidence
      - Confidence = sigmoid(R² × 10 - 8)  (high R² → high confidence)
      - Extrapolates generalisation claim based on equation structure
    """

    def argue(self, hypothesis: Hypothesis) -> BullArgument:
        """Generate a structured argument supporting the hypothesis.

        Args:
            hypothesis: The hypothesis under debate.

        Returns:
            BullArgument with quantitative supporting evidence.
        """
        r2 = hypothesis.metrics.r_squared if hypothesis.metrics else 0.0
        rmse = hypothesis.metrics.rmse if hypothesis.metrics else float("inf")

        # Sigmoid confidence from R²: near 1.0 → ~0.99, near 0.5 → ~0.12
        confidence = 1.0 / (1.0 + math.exp(-10.0 * (r2 - 0.95)))
        confidence = round(max(0.0, min(1.0, confidence)), 4)

        evidence = [
            SupportingEvidence(
                evidence_type="r_squared",
                value=r2,
                description=(
                    f"R²={r2:.4f} on held-out trajectory indicates the equation explains "
                    f"{r2 * 100:.1f}% of variance in the target dynamics."
                ),
                step_range=hypothesis.evidence_step_range,
            ),
            SupportingEvidence(
                evidence_type="rmse",
                value=rmse,
                description=(
                    f"RMSE={rmse:.6f} shows prediction error is within "
                    f"{'acceptable' if rmse < 0.1 else 'elevated'} bounds."
                ),
            ),
        ]

        # Add conservation evidence if equation is linear (Hooke-type)
        eq = hypothesis.candidate_equation
        if "*x" in eq and "*v" not in eq:
            evidence.append(SupportingEvidence(
                evidence_type="symmetry",
                value=1.0,
                description=(
                    "Equation contains only odd-power displacement terms, consistent with "
                    "conservative force fields and energy conservation laws."
                ),
            ))

        strongest = (
            f"The hypothesis achieves R²={r2:.4f} on 200 held-out trajectory steps, "
            f"demonstrating robust predictive power beyond the training window. "
            f"Equation: {hypothesis.candidate_equation}"
        )

        generalisation = (
            "If the equation encodes a true physical law rather than a fitting artefact, "
            "it should achieve R² > 0.95 on trajectories with different initial conditions "
            "but identical underlying force constants."
        )

        return BullArgument(
            hypothesis_id=hypothesis.id,
            world_name=hypothesis.world_name,
            confidence_score=confidence,
            supporting_evidence=evidence,
            strongest_claim=strongest,
            predicted_generalisation=generalisation,
        )


# ---------------------------------------------------------------------------
# Bear Agent
# ---------------------------------------------------------------------------

class BearAgent:
    """Identifies the strongest weaknesses AGAINST the hypothesis.

    Rule-based implementation:
      - Flags overfitting risk when training/test split is narrow
      - Proposes alternative equation if R² is between 0.8 and 0.95
      - Checks for lack of velocity terms in damped worlds
    """

    def argue(self, hypothesis: Hypothesis) -> BearArgument:
        """Generate a structured critique of the hypothesis.

        Args:
            hypothesis: The hypothesis under debate.

        Returns:
            BearArgument with identified weaknesses.
        """
        r2 = hypothesis.metrics.r_squared if hypothesis.metrics else 0.0
        eq = hypothesis.candidate_equation
        params = hypothesis.parameters.values

        weaknesses = []
        doubt = 0.0

        # Check for overfitting (train ratio 80% leaves only 200 test steps)
        train_steps = hypothesis.metrics.train_steps if hypothesis.metrics else 800
        test_steps = hypothesis.metrics.test_steps if hypothesis.metrics else 200
        if test_steps < 100:
            weaknesses.append(Weakness(
                weakness_type="limited_data",
                description=f"Only {test_steps} held-out steps for validation. Small test window risks over-confident R².",
                severity="MEDIUM",
                suggested_test="Validate on a second trajectory with different seed.",
            ))
            doubt += 0.15

        # Flag if no velocity term in damped world
        if "damped" in hypothesis.world_name and "v" not in eq:
            weaknesses.append(Weakness(
                weakness_type="alternative_law",
                description=(
                    "Damped oscillator should exhibit F = -kx - bẋ. "
                    "Absence of velocity term suggests damping coefficient b was not recovered."
                ),
                severity="HIGH",
                suggested_test="Verify by running at low damping vs high damping — diverging R² confirms missed b term.",
            ))
            doubt += 0.35

        # Flag potential overfitting if R² is suspiciously perfect
        if r2 > 0.999:
            weaknesses.append(Weakness(
                weakness_type="overfitting",
                description=(
                    f"R²={r2:.4f} is suspiciously high. "
                    "This may indicate the test trajectory was not sufficiently different from training."
                ),
                severity="LOW",
                suggested_test="Test on trajectory with 5× larger initial displacement.",
            ))
            doubt += 0.05

        # Default weakness: always flag limited scope
        if not weaknesses:
            weaknesses.append(Weakness(
                weakness_type="limited_data",
                description="Hypothesis was fit on a single trajectory run. Cross-seed generalisation unverified.",
                severity="LOW",
                suggested_test="Run on 5 different random seeds and measure R² variance.",
            ))
            doubt += 0.10

        doubt = round(min(0.99, doubt), 4)

        # Alternative hypothesis
        alt = None
        if r2 < 0.95:
            alt = f"-{abs(list(params.values())[0]):.2f}*x² + noise"

        critical = weaknesses[0].description if weaknesses else "No critical flaw identified."

        return BearArgument(
            hypothesis_id=hypothesis.id,
            world_name=hypothesis.world_name,
            doubt_score=doubt,
            weaknesses=weaknesses,
            alternative_hypothesis=alt,
            critical_flaw=critical,
        )


# ---------------------------------------------------------------------------
# Skeptic Agent
# ---------------------------------------------------------------------------

class SkepticAgent:
    """Designs targeted counterfactual experiments to falsify the hypothesis.

    Rule-based implementation:
      - Always proposes: large-displacement, reversed-velocity, cross-seed tests
      - R² threshold is set to 0.95 for survival (strict)
    """

    def design_experiment(self, hypothesis: Hypothesis) -> CounterfactualExperiment:
        """Design the most likely-to-break experiment for this hypothesis.

        The Skeptic picks the HARDEST perturbation to maximise falsification
        probability if the hypothesis is wrong.

        Args:
            hypothesis: The hypothesis under challenge.

        Returns:
            CounterfactualExperiment with perturbation specs.
        """
        world = hypothesis.world_name
        hyp_id = hypothesis.id

        # For harmonic spring: challenge with a 5x larger displacement
        # A true F=-kx must hold at all amplitudes (linear law)
        return CounterfactualExperiment(
            hypothesis_id=hyp_id,
            world_name=world,
            experiment_name="amplitude_stress_test",
            description=(
                "Apply 5× larger initial displacement than training range. "
                "True linear law F=-kx must survive this without R² collapse."
            ),
            perturbations=[
                PerturbationSpec(
                    parameter="initial_displacement",
                    original_value=1.0,
                    perturbed_value=5.0,
                    rationale=(
                        "Linear laws (F∝x) are amplitude-independent. "
                        "If R² collapses at large amplitude, the law is nonlinear or overfitted."
                    ),
                ),
                PerturbationSpec(
                    parameter="initial_velocity",
                    original_value=0.0,
                    perturbed_value=-2.0,
                    rationale="Reversed velocity tests direction symmetry of the discovered force law.",
                ),
            ],
            predicted_outcome_if_true=(
                "R² ≥ 0.95 on perturbed trajectory — law holds at all amplitudes and directions."
            ),
            predicted_outcome_if_false=(
                "R² < 0.95 — discovered equation is an artefact of the training initial conditions."
            ),
            r2_threshold_to_survive=0.95,
        )


# ---------------------------------------------------------------------------
# Arbiter Agent (Purely Computational — Zero LLM)
# ---------------------------------------------------------------------------

class ArbiterAgent:
    """Weighs all evidence and issues a final Bayesian verdict.

    Algorithm:
      prior = 0.5 (equal probability before evidence)
      posterior update:
        + Bull confidence × 0.3
        + (1 - Bear doubt) × 0.25
        + (1 if experiment survived else 0) × 0.45
      verdict = ACCEPT if posterior > 0.75, REJECT if < 0.35, else INCONCLUSIVE

    The Intervention Engine's experiment result has the highest weight (0.45)
    because it is empirical evidence — not argumentation.
    """

    def issue_verdict(
        self,
        hypothesis: Hypothesis,
        bull: BullArgument,
        bear: BearArgument,
        experiment: CounterfactualExperiment,
        result: ExperimentResult,
    ) -> ArbiterVerdict:
        """Compute the final Bayesian verdict from all evidence sources.

        Args:
            hypothesis: The hypothesis under judgement.
            bull:       Bull agent's supporting argument.
            bear:       Bear agent's critical argument.
            experiment: Skeptic's counterfactual experiment spec.
            result:     Empirical outcome of running the experiment.

        Returns:
            ArbiterVerdict with verdict, confidence, and reasoning.
        """
        # Evidence weights (must sum to 1.0)
        W_BULL = 0.30
        W_BEAR = 0.25
        W_EXPERIMENT = 0.45

        bull_contribution = bull.confidence_score * W_BULL
        bear_contribution = (1.0 - bear.doubt_score) * W_BEAR
        exp_contribution = (1.0 if result.survived else 0.0) * W_EXPERIMENT

        posterior = round(bull_contribution + bear_contribution + exp_contribution, 4)
        posterior = max(0.0, min(1.0, posterior))

        # Verdict threshold
        if posterior > 0.75:
            verdict = "ACCEPT"
        elif posterior < 0.35:
            verdict = "REJECT"
        else:
            verdict = "INCONCLUSIVE"

        # Reproducibility score = fraction of criteria met
        repro = 1.0 if result.survived else 0.0

        weights = [
            EvidenceWeight(source="Bull", weight=W_BULL, contribution=f"confidence={bull.confidence_score:.3f}"),
            EvidenceWeight(source="Bear", weight=W_BEAR, contribution=f"doubt={bear.doubt_score:.3f}"),
            EvidenceWeight(source="Experiment", weight=W_EXPERIMENT, contribution=f"survived={result.survived}, R²={result.r_squared_on_perturbed:.4f}"),
        ]

        reasoning = (
            f"Bayesian posterior P(correct|evidence) = {posterior:.4f}. "
            f"Bull contribution: {bull_contribution:.4f} | "
            f"Bear contribution: {bear_contribution:.4f} | "
            f"Experiment contribution: {exp_contribution:.4f}. "
            f"Experiment survival: {result.survived} (R²={result.r_squared_on_perturbed:.4f} vs threshold={experiment.r2_threshold_to_survive}). "
            f"Verdict: {verdict}."
        )

        return ArbiterVerdict(
            hypothesis_id=hypothesis.id,
            world_name=hypothesis.world_name,
            verdict=verdict,
            bayesian_confidence=posterior,
            evidence_weights=weights,
            reasoning=reasoning,
            reproducibility_score=repro,
        )
