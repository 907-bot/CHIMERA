"""Self-Assembling Vesicular Compartment Engine (CHIMERA v4.0 - Phase 12)

Simulates protocell membrane self-assembly, nutrient uptake through semi-permeable lipid bilayers,
and vesicle fission/growth.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from packages.abiogenesis.models import VesicleMembrane


class VesicleCompartmentEngine:
    """Manages protocellular vesicles, selective permeability, internal reactions, and growth."""

    def __init__(self, critical_micelle_concentration: float = 10.0):
        self.cmc = critical_micelle_concentration

    def assemble_vesicle(
        self,
        vesicle_id: str,
        lipid_pool: float,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Optional[VesicleMembrane]:
        """Self-assembles lipid bilayer sphere if lipid pool exceeds critical micelle concentration (CMC)."""
        if lipid_pool < self.cmc:
            return None

        # Radius scales with sqrt of lipid count: R ∝ sqrt(N)
        radius = float(np.sqrt(lipid_pool)) * 0.5
        return VesicleMembrane(
            vesicle_id=vesicle_id,
            center=center,
            radius=radius,
            lipid_count=int(lipid_pool),
            internal_metabolites={"substrate": 1.0, "catalyst": 0.1},
            permeability=0.3,
        )

    def step_metabolism(
        self,
        vesicle: VesicleMembrane,
        external_substrate: float,
        dt: float = 0.05,
    ) -> VesicleMembrane:
        """Simulates nutrient diffusion into vesicle, internal catalyzed synthesis of new lipids, and vesicle growth."""
        internal_sub = vesicle.internal_metabolites.get("substrate", 0.0)
        internal_cat = vesicle.internal_metabolites.get("catalyst", 0.0)

        # 1. Diffusion flux: J = P * (S_ext - S_int)
        flux = vesicle.permeability * (external_substrate - internal_sub)
        sub_new = max(0.0, internal_sub + flux * dt)

        # 2. Catalytic production of new membrane lipids: dL/dt = k * [substrate] * [catalyst]
        lipid_synth_rate = 0.5 * sub_new * internal_cat
        lipid_delta = lipid_synth_rate * dt
        new_lipid_count = vesicle.lipid_count + int(np.round(lipid_delta))
        new_radius = float(np.sqrt(new_lipid_count)) * 0.5

        # Substrate consumption
        sub_new = max(0.0, sub_new - lipid_delta)

        new_metabolites = {
            "substrate": sub_new,
            "catalyst": internal_cat,
        }

        return vesicle.model_copy(
            update={
                "lipid_count": new_lipid_count,
                "radius": new_radius,
                "internal_metabolites": new_metabolites,
            }
        )
