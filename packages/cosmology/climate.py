"""Latitudinal Planetary Climate Model (CHIMERA v5.0 - Phase 13)

Implements Budyko-Sellers 1D Energy Balance Climate Model (EBM):
    C * ∂T/∂t = Q * S(y) * (1 - α(T)) - (A + B*T) + D * ∂²T/∂y² + ΔF_CO2
where:
    y = sin(latitude)
    α(T) = ice-albedo feedback (higher albedo below freezing)
    A + B*T = Outgoing Longwave Radiation (OLR)
    D = Meridional heat transport coefficient
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
from packages.cosmology.models import ClimateGridState


class PlanetaryClimateModel:
    """1D latitudinal energy balance climate model."""

    def __init__(
        self,
        num_zones: int = 18,
        solar_constant: float = 1361.0,
        heat_capacity: float = 1e7,  # J / (m^2 K)
        diffusivity_d: float = 0.6,
        dt_years: float = 0.1,
    ):
        self.num_zones = num_zones
        self.solar_constant = solar_constant
        self.c_heat = heat_capacity
        self.D = diffusivity_d
        self.dt = dt_years

        # Discretize latitudes from -85 to +85 degrees
        self.lats = np.linspace(-85, 85, num_zones)
        self.sin_lats = np.sin(np.radians(self.lats))
        self.dy = 2.0 / num_zones

        # Solar distribution function S(y) approx 1 - 0.482 * (3y² - 1)/2
        p2 = 0.5 * (3.0 * (self.sin_lats ** 2) - 1.0)
        self.s_dist = 1.0 - 0.482 * p2

        # Empirical OLR constants (A, B)
        self.A_olr = 204.0  # W/m^2
        self.B_olr = 2.17   # W/(m^2 K)

    def initialize_state(self, mean_temp_celsius: float = 15.0) -> ClimateGridState:
        """Initialize climate state across latitudes."""
        temp_k = (mean_temp_celsius + 273.15) - 30.0 * (self.sin_lats ** 2)
        ice_cov = np.where(temp_k < 263.15, 1.0, 0.0)

        return ClimateGridState(
            step=0,
            time=0.0,
            latitudes=tuple(float(l) for l in self.lats),
            temperatures=tuple(float(t) for t in temp_k),
            ice_coverage=tuple(float(i) for i in ice_cov),
            co2_ppm=280.0,
            solar_constant=self.solar_constant,
        )

    def step(self, state: ClimateGridState) -> ClimateGridState:
        """Advance planetary climate state by one timestep dt."""
        T = np.array(state.temperatures, dtype=np.float64)

        # 1. Ice-albedo feedback: α = 0.62 if T < -10°C, 0.30 if T > 0°C, linear interpolation between
        albedo = np.where(T < 263.15, 0.62, np.where(T > 273.15, 0.30, 0.30 + 0.32 * (273.15 - T) / 10.0))

        # 2. Absorbed solar radiation: Q = (S0 / 4) * S(y) * (1 - α)
        insolation = (state.solar_constant / 4.0) * self.s_dist * (1.0 - albedo)

        # 3. Outgoing Longwave Radiation: OLR = A + B * (T - 273.15) - 5.35 * ln(CO2 / 280)
        co2_forcing = 5.35 * np.log(max(state.co2_ppm, 1.0) / 280.0)
        olr = self.A_olr + self.B_olr * (T - 273.15) - co2_forcing

        # 4. Meridional heat diffusion: D * (T_mean - T)
        t_mean = np.mean(T)
        meridional_transport = self.D * (t_mean - T)

        # Temperature rate of change (scaled to annual response)
        dT_dt = (insolation - olr + meridional_transport) * 0.05
        T_next = T + self.dt * dT_dt
        ice_next = np.where(T_next < 263.15, 1.0, 0.0)

        return ClimateGridState(
            step=state.step + 1,
            time=state.time + self.dt,
            latitudes=state.latitudes,
            temperatures=tuple(float(t) for t in T_next),
            ice_coverage=tuple(float(i) for i in ice_next),
            co2_ppm=state.co2_ppm,
            solar_constant=state.solar_constant,
        )
