# -*- coding: utf-8 -*-
"""Tests for :class:`mtpy.modeling.femtic.mesh.FemticMesh`.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtpy.modeling.femtic.mesh import FemticMesh


@pytest.fixture
def mesh(station_frame: pd.DataFrame) -> FemticMesh:
    """A FemticMesh built from the shared station frame."""
    return FemticMesh(station_frame, start_res=100.0)


class TestFemticMeshConstruction:
    """Construction, validation, and alternative constructors."""

    def test_requires_core_columns(self):
        with pytest.raises(KeyError, match="missing required column"):
            FemticMesh(pd.DataFrame({"station": ["a"]}))

    def test_copies_input_frame(self, station_frame: pd.DataFrame):
        mesh = FemticMesh(station_frame)
        mesh.mt_df.loc[0, "east"] = -999.0
        assert station_frame.loc[0, "east"] == 0.0

    def test_start_res_cast_to_float(self, station_frame: pd.DataFrame):
        mesh = FemticMesh(station_frame, start_res=50)
        assert isinstance(mesh.start_res, float)
        assert mesh.start_res == 50.0

    def test_from_mt_dataframe(self, station_frame: pd.DataFrame):
        mesh = FemticMesh.from_mt_dataframe(station_frame, start_res=10.0)
        assert isinstance(mesh, FemticMesh)
        assert mesh.start_res == 10.0

    def test_from_mt_data(self, station_frame: pd.DataFrame):
        class _FakeMTData:
            def to_dataframe(self_inner):
                return station_frame

        mesh = FemticMesh.from_mt_data(_FakeMTData())
        assert mesh.n_stations == 3

    def test_has_logger(self, mesh: FemticMesh):
        # mtpy-v2 modeling classes carry a loguru logger instance.
        assert hasattr(mesh, "logger")


class TestFemticMeshAccessors:
    """Station / period accessors."""

    def test_stations_first_seen_order(self, mesh: FemticMesh):
        assert list(mesh.stations) == ["a", "b", "c"]

    def test_n_stations(self, mesh: FemticMesh):
        assert mesh.n_stations == 3

    def test_periods_sorted_unique(self, mesh: FemticMesh):
        assert np.array_equal(mesh.periods, np.array([0.01, 1.0]))

    def test_n_periods(self, mesh: FemticMesh):
        assert mesh.n_periods == 2

    def test_periods_empty_without_column(self, station_frame: pd.DataFrame):
        mesh = FemticMesh(station_frame.drop(columns=["period"]))
        assert mesh.periods.size == 0
        assert mesh.n_periods == 0

    def test_station_coords_one_row_per_station(self, mesh: FemticMesh):
        coords = mesh.station_coords_m
        assert list(coords.columns) == ["station", "east", "north"]
        assert len(coords) == 3


class TestSkinDepth:
    """Skin-depth-based depth sizing."""

    def test_max_period_deeper_than_min(self, mesh: FemticMesh):
        assert mesh.skin_depth_km("max") > mesh.skin_depth_km("min")

    def test_known_value(self, mesh: FemticMesh):
        # 500 * sqrt(100 * 1.0) / 1000 == 5.0 km
        assert mesh.skin_depth_km("max") == pytest.approx(5.0)

    def test_respects_res_override(self, mesh: FemticMesh):
        base = mesh.skin_depth_km("max")
        # Quadrupling resistivity doubles the skin depth.
        assert mesh.skin_depth_km("max", res=400.0) == pytest.approx(2 * base)

    def test_invalid_which_raises(self, mesh: FemticMesh):
        with pytest.raises(ValueError, match="'max' or 'min'"):
            mesh.skin_depth_km("sideways")

    def test_missing_period_raises(self, station_frame: pd.DataFrame):
        mesh = FemticMesh(station_frame.drop(columns=["period"]))
        with pytest.raises(KeyError, match="period"):
            mesh.skin_depth_km("max")


class TestDunders:
    def test_write_inputs_not_implemented(self, mesh: FemticMesh, tmp_path):
        with pytest.raises(NotImplementedError):
            mesh.write_inputs(tmp_path)

    def test_repr_mentions_counts(self, mesh: FemticMesh):
        text = repr(mesh)
        assert "FemticMesh" in text
        assert "n_stations=3" in text
        assert "n_periods=2" in text
