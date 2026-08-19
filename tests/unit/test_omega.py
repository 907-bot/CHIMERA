"""Unit Tests for Phase 18: The Omega Observatory (CHIMERA v10.0)"""

import pytest
from packages.omega.models import RealityRecord, RealityCatalogQuery
from packages.omega.realities_db import UniversalRealitiesDatabase
from packages.omega.stream_pipeline import OmegaDiscoveryStreamPipeline
from packages.omega.portal import OmegaPortalEngine


def test_realities_db_query():
    db = UniversalRealitiesDatabase()
    r1 = RealityRecord(
        reality_id="real_001",
        seed=42,
        dimension_domain="quantum",
        discovered_equations=["i*hbar*d_psi/dt = H*psi"],
        emergence_metrics={"complexity": 0.9},
    )
    r2 = RealityRecord(
        reality_id="real_002",
        seed=101,
        dimension_domain="abiogenesis",
        discovered_equations=["dx/dt = k*x*y - phi*x"],
        emergence_metrics={"complexity": 0.4},
    )

    db.insert_reality(r1)
    db.insert_reality(r2)

    assert db.total_realities_indexed == 2

    # Query domain
    res_domain = db.query(RealityCatalogQuery(domain="quantum"))
    assert len(res_domain) == 1
    assert res_domain[0].reality_id == "real_001"

    # Query complexity
    res_high_comp = db.query(RealityCatalogQuery(min_emergence_score=0.8))
    assert len(res_high_comp) == 1


def test_omega_portal_json_export():
    r = RealityRecord(
        reality_id="real_001",
        seed=42,
        dimension_domain="cosmology",
        physical_constants={"G": 1.5},
        emergence_metrics={"complexity": 0.8, "diversity": 0.7},
    )
    json_export = OmegaPortalEngine.export_reality_manifold_json([r])
    assert json_export["total_nodes"] == 1
    assert json_export["manifold_points"][0]["position"] == [1.5, 0.8, 0.7]
