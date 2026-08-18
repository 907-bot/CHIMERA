# CHIMERA Research & Scientific Reproducibility Protocol

## 1. Reproducibility Guarantee

Every experiment published by CHIMERA MUST specify:
- `seed`: Integer seed for the pseudo-random number generator.
- `world_config`: Complete JSON/YAML configuration file detailing dimensions, parameters, particles, and force constants.
- `code_version`: Git commit SHA matching the exact codebase used for execution.

Given these three parameters, execution on any compliant environment MUST yield a bitwise identical final state and trajectory hash.

## 2. Benchmark Validation Metrics

Physical law recovery is evaluated using strict quantitative metrics rather than qualitative textual assertions:
- **Prediction Error ($MSE$)**: Mean Squared Error on held-out test trajectories.
- **Coefficient of Determination ($R^2$)**: Trajectory fit accuracy ($R^2 > 0.99$ required for baseline law confirmation).
- **Parameter Drift Error**: Error percentage between discovered parameters $\hat{\theta}$ and true withheld parameters $\theta^*$:
  $$\text{Error}_\theta = \frac{|\hat{\theta} - \theta^*|}{|\theta^*|} \times 100\% < 1.0\%$$
- **Energy Conservation Drift**: Energy change ratio over simulation duration $T$:
  $$\frac{|\Delta E|}{E_0} = \frac{|E(T) - E(0)|}{E(0)} < 10^{-4}$$
