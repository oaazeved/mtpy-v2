# -*- coding: utf-8 -*-
"""Tests for :class:`mtpy.modeling.femtic.hexmesh.DeformableHexMesh`.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path

The GIS-backed topography path (geopandas/rasterio/shapely) is not
exercised here; those tests are skipped unless the optional dependencies
are present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtpy.modeling.femtic.hexmesh import DeformableHexMesh


@pytest.fixture
def hexmesh(station_frame: pd.DataFrame) -> DeformableHexMesh:
    """A small DeformableHexMesh with reduced padding/sub-levels."""
    return DeformableHexMesh(station_frame, n_padding=4, n_sub_levels=5)


class TestConstruction:
    def test_is_femtic_mesh(self, hexmesh: DeformableHexMesh):
        from mtpy.modeling.femtic.mesh import FemticMesh

        assert isinstance(hexmesh, FemticMesh)

    def test_initial_resistivity_sets_start_res(self, station_frame):
        mesh = DeformableHexMesh(station_frame, initial_resistivity=33.0)
        assert mesh.start_res == 33.0
        assert mesh.initial_resistivity == 33.0

    def test_inherits_logger(self, hexmesh: DeformableHexMesh):
        assert hasattr(hexmesh, "logger")


class TestVertices:
    """Computed mesh-vertex coordinate arrays."""

    def test_horizontal_vertices_monotonic(self, hexmesh, subtests):
        with subtests.test("x_increasing"):
            assert np.all(np.diff(hexmesh.x_vertices) > 0)
        with subtests.test("y_increasing"):
            assert np.all(np.diff(hexmesh.y_vertices) > 0)

    def test_z_vertices_present_and_increasing(self, hexmesh):
        z = hexmesh.z_vertices
        assert z.size > 0
        assert np.all(np.diff(z) > 0)

    def test_log_zmethod_runs(self, station_frame):
        mesh = DeformableHexMesh(
            station_frame, zmethod="log", n_padding=4, n_sub_levels=5
        )
        assert mesh.z_vertices.size > 0

    def test_n_cells_matches_vertex_counts(self, hexmesh):
        nx, ny, nz = hexmesh.n_cells
        assert nx == len(hexmesh.x_vertices) - 1
        assert ny == len(hexmesh.y_vertices) - 1
        assert nz == len(hexmesh.z_vertices) - 1


class TestWriteInputs:
    """File writers that do not require the GIS stack."""

    def test_write_meshgen_inp_roundtrips(self, hexmesh, tmp_path, subtests):
        path = tmp_path / "meshgen.inp"
        hexmesh.write_meshgen_inp(path)
        assert path.exists()

        parsed = DeformableHexMesh.read_meshgen_inp(path)
        with subtests.test("has_division_numbers"):
            assert "DIVISION_NUMBERS" in parsed
        with subtests.test("has_coordinate_blocks"):
            assert "X_COORDINATES" in parsed
            assert "Y_COORDINATES" in parsed
            assert "Z_COORDINATES" in parsed
        with subtests.test("has_resistivity_blocks"):
            assert "INITIAL_RESISTIVITY" in parsed
            assert "AIR_RESISTIVITY" in parsed

    def test_write_obs_site(self, hexmesh, tmp_path):
        hexmesh.write_obs_site(tmp_path)
        obs = tmp_path / "obs_site.dat"
        assert obs.exists()
        # First line is the station count.
        assert obs.read_text().splitlines()[0].strip() == str(hexmesh.n_stations)

    def test_write_inputs_creates_directory(self, hexmesh, tmp_path):
        out = tmp_path / "nested" / "run"
        hexmesh.write_inputs(out)
        assert (out / "meshgen.inp").exists()
        assert (out / "obs_site.dat").exists()
