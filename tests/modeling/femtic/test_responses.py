# -*- coding: utf-8 -*-
"""Tests for :mod:`mtpy.modeling.femtic.responses`.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path

Covers the ``observe.dat`` reader/writer round-trip, the module-level
unit/convention/error-floor helpers, and the convergence reader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mtpy.modeling.femtic.responses import (
    FemticConvergence,
    FemticData,
    NODATA_VAL,
    OHM_TO_MVKMNT,
    apply_error_floor,
    conjugate,
    convert_units,
    drop_nodata,
    prep_data,
)


class TestFemticData:
    """Parsing and round-tripping a FEMTIC ``observe.dat``."""

    def test_from_file_counts(self, observe_dat, subtests):
        data = FemticData.from_file(observe_dat)
        with subtests.test("n_stations"):
            assert data.n_stations == 2
        with subtests.test("n_periods"):
            assert data.n_periods == 2
        with subtests.test("default_units_ohm"):
            assert data.units == "ohm"
        with subtests.test("default_convention_plus"):
            # from_file flips imaginary signs to mtpy's exp(+iwt) convention.
            assert data.phase_convention == "+"

    def test_imaginary_sign_flipped_on_read(self, observe_dat):
        data = FemticData.from_file(observe_dat)
        df = data.to_dataframe()
        # File wrote Im(Zxx) = -0.1 under exp(-iwt); read flips it to +0.1.
        assert df["Im(Zxx)"].iloc[0] == pytest.approx(0.1)

    def test_to_file_roundtrip(self, observe_dat, tmp_path, subtests):
        data = FemticData.from_file(observe_dat)
        out = tmp_path / "observe_out.dat"
        data.to_file(out)
        assert out.exists()

        reread = FemticData.from_file(out)
        with subtests.test("stations_preserved"):
            assert reread.n_stations == data.n_stations
        with subtests.test("periods_preserved"):
            assert reread.n_periods == data.n_periods
        with subtests.test("values_preserved"):
            a = data.to_dataframe()["Re(Zxy)"].to_numpy()
            b = reread.to_dataframe()["Re(Zxy)"].to_numpy()
            assert np.allclose(a, b)

    def test_len(self, observe_dat):
        data = FemticData.from_file(observe_dat)
        # Two stations x two periods = four records.
        assert len(data) == 4


class TestConvertUnits:
    def test_divides_by_constant(self, impedance_frame):
        out = convert_units(impedance_frame, data_type="Z")
        ratio = impedance_frame["z_xy"].iloc[0] / out["z_xy"].iloc[0]
        assert ratio == pytest.approx(OHM_TO_MVKMNT)

    def test_does_not_mutate_input(self, impedance_frame):
        before = impedance_frame["z_xy"].iloc[0]
        convert_units(impedance_frame, data_type="Z")
        assert impedance_frame["z_xy"].iloc[0] == before

    def test_non_z_passthrough(self, impedance_frame):
        out = convert_units(impedance_frame, data_type="AR")
        assert out["z_xy"].iloc[0] == impedance_frame["z_xy"].iloc[0]


class TestConjugate:
    def test_z_conjugated(self, impedance_frame):
        out = conjugate(impedance_frame, data_type="Z")
        assert out["z_xy"].iloc[0] == np.conj(impedance_frame["z_xy"].iloc[0])

    def test_vtf_conjugated(self, impedance_frame):
        out = conjugate(impedance_frame, data_type="VTF")
        assert out["t_zx"].iloc[0] == np.conj(impedance_frame["t_zx"].iloc[0])

    def test_does_not_mutate_input(self, impedance_frame):
        before = impedance_frame["z_xy"].iloc[0]
        conjugate(impedance_frame, data_type="Z")
        assert impedance_frame["z_xy"].iloc[0] == before


class TestApplyErrorFloor:
    def test_adds_split_error_columns(self, impedance_frame):
        out = apply_error_floor(impedance_frame, data_type="Z", error_floor_Z=0.05)
        for comp in ("z_xx", "z_xy", "z_yx", "z_yy"):
            assert f"{comp}_R_error" in out.columns
            assert f"{comp}_I_error" in out.columns

    def test_floor_is_geometric_mean_of_offdiag(self, impedance_frame):
        out = apply_error_floor(impedance_frame, data_type="Z", error_floor_Z=0.05)
        zxy = abs(impedance_frame["z_xy"].iloc[0])
        zyx = abs(impedance_frame["z_yx"].iloc[0])
        expected = 0.05 * np.sqrt(zxy * zyx)
        assert out["z_xx_R_error"].iloc[0] == pytest.approx(expected)


class TestDropNodata:
    """Per-frequency nodata screening (femticPy-style row dropping)."""

    def test_clean_frame_unchanged(self, impedance_frame):
        out = drop_nodata(impedance_frame, data_type="Z")
        assert len(out) == len(impedance_frame)

    def test_nan_value_drops_row(self, impedance_frame):
        df = impedance_frame.copy()
        df.loc[0, "z_xy"] = np.nan
        out = drop_nodata(df, data_type="Z")
        # The whole station/period record is dropped, not just z_xy.
        assert len(out) == len(impedance_frame) - 1
        assert out["period"].tolist() == [1.0]

    def test_nan_error_drops_row(self, impedance_frame):
        df = impedance_frame.copy()
        df.loc[1, "z_yx_error"] = np.nan
        out = drop_nodata(df, data_type="Z")
        assert len(out) == len(impedance_frame) - 1
        assert out["period"].tolist() == [0.01]

    def test_sentinel_value_drops_row(self, impedance_frame):
        df = impedance_frame.copy()
        df.loc[0, "z_yy"] = NODATA_VAL
        out = drop_nodata(df, data_type="Z")
        assert len(out) == len(impedance_frame) - 1

    def test_does_not_mutate_input(self, impedance_frame):
        df = impedance_frame.copy()
        df.loc[0, "z_xy"] = np.nan
        before = len(df)
        drop_nodata(df, data_type="Z")
        assert len(df) == before
        assert np.isnan(df.loc[0, "z_xy"])

    def test_resets_index(self, impedance_frame):
        df = impedance_frame.copy()
        df.loc[0, "z_xy"] = np.nan
        out = drop_nodata(df, data_type="Z")
        assert list(out.index) == list(range(len(out)))

    def test_vtf_screens_tipper_columns(self, impedance_frame):
        df = impedance_frame.copy()
        df.loc[0, "t_zx"] = np.nan
        # A NaN tipper is ignored when only impedance is screened ...
        assert len(drop_nodata(df, data_type="Z")) == len(impedance_frame)
        # ... but drops the row when the VTF family is screened.
        assert len(drop_nodata(df, data_type="VTF")) == len(impedance_frame) - 1

    def test_prep_data_drops_nodata_rows(self, impedance_frame):
        df = impedance_frame.copy()
        df.loc[0, "z_xy"] = np.nan
        out = prep_data(df, data_type="Z", error_floor_Z=0.05)
        # The NaN record is screened before unit-conversion/error-floor.
        assert len(out) == len(impedance_frame) - 1
        assert out["period"].tolist() == [1.0]


class TestFemticConvergence:
    def test_from_file(self, convergence_file, subtests):
        cnv = FemticConvergence.from_file(convergence_file)
        with subtests.test("final_rms"):
            # Last kept row (Iter# 2, Retrial# 1) after dropping lower retrials.
            assert cnv.final_rms == pytest.approx(2.0)
        with subtests.test("best_rms"):
            assert cnv.best_rms == pytest.approx(2.0)
        with subtests.test("best_iteration"):
            assert cnv.best_iteration == 2

    def test_to_dataframe_returns_copy(self, convergence_file):
        cnv = FemticConvergence.from_file(convergence_file)
        df = cnv.to_dataframe()
        df.loc[0, "RMS"] = -1.0
        assert cnv.to_dataframe().loc[0, "RMS"] != -1.0
