# -*- coding: utf-8 -*-
"""Tests for :class:`mtpy.modeling.femtic.tetramesh.TetraMesh`.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path

Only the non-GIS file writers are exercised (no topography, no coast
derivation), so geopandas/rasterio/shapely are not required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mtpy.modeling.femtic.tetramesh import TetraMesh


@pytest.fixture
def tetramesh(station_frame: pd.DataFrame) -> TetraMesh:
    return TetraMesh(station_frame, start_res=100.0)


class TestConstruction:
    def test_is_femtic_mesh(self, tetramesh: TetraMesh):
        from mtpy.modeling.femtic.mesh import FemticMesh

        assert isinstance(tetramesh, FemticMesh)

    def test_inherits_logger(self, tetramesh: TetraMesh):
        assert hasattr(tetramesh, "logger")

    def test_unknown_config_key_warns(self, station_frame: pd.DataFrame):
        with pytest.warns(UserWarning, match="unrecognised config keys"):
            TetraMesh(station_frame, not_a_real_key=1)


class TestModelFrame:
    def test_model_center_is_north_east(self, tetramesh: TetraMesh):
        # bbox center of the station frame: north in (0..1000), east in (0..2000)
        north0, east0 = tetramesh.model_center
        assert north0 == pytest.approx(500.0)
        assert east0 == pytest.approx(1000.0)

    def test_centered_dataframe_is_centered(self, tetramesh: TetraMesh):
        cdf = tetramesh.centered_dataframe
        # After recentring, the bbox midpoints sit at the origin.
        assert cdf["north"].mean() == pytest.approx(0.0, abs=1e-6)
        assert cdf["east"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_bounds_km_has_six_components(self, tetramesh: TetraMesh):
        bounds = tetramesh.bounds_km
        assert len(bounds) == 6


class TestWriteInputs:
    """End-to-end file writing without the topography pipeline."""

    def test_writes_expected_files(self, tetramesh: TetraMesh, tmp_path, subtests):
        written = tetramesh.write_inputs(tmp_path)
        expected = {
            "analysis_domain.dat",
            "control.dat",
            "coast_line.dat",
            "makeMtr.param",
            "obs_site.dat",
            "observing_site.dat",
            "resistivity_attr.dat",
        }
        with subtests.test("returns_paths"):
            assert expected.issubset(set(written))
        with subtests.test("files_on_disk"):
            for name in expected:
                assert (tmp_path / name).exists()

    def test_creates_missing_directory(self, tetramesh: TetraMesh, tmp_path):
        out = tmp_path / "deep" / "run"
        tetramesh.write_inputs(out)
        assert (out / "control.dat").exists()
