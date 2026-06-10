# -*- coding: utf-8 -*-
"""Tests for :mod:`mtpy.modeling.femtic.modem`.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path

The pyvista-backed exports (3-D grid, VTK, slices) are skipped unless
pyvista is installed; the model-file reader/writer needs no optional
dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.modeling.femtic.modem import ModEMModel


class TestModEMModelIO:
    """ModEM rectilinear model-file reading and round-tripping."""

    def test_from_file_shape(self, modem_model_file, subtests):
        model = ModEMModel.from_file(modem_model_file)
        with subtests.test("shape"):
            assert model.shape == (2, 2, 2)
        with subtests.test("n_air"):
            assert model.n_air == 0
        with subtests.test("valtype"):
            assert model.valtype == "LOGE"

    def test_has_logger(self, modem_model_file):
        model = ModEMModel.from_file(modem_model_file)
        assert hasattr(model, "logger")

    def test_roundtrip_preserves_data(self, modem_model_file, tmp_path):
        model = ModEMModel.from_file(modem_model_file)
        out = tmp_path / "model_out.rho"
        model.to_file(out)
        assert out.exists()

        reread = ModEMModel.from_file(out)
        assert np.array_equal(model.data, reread.data)

    def test_to_femtic_meshgen(self, modem_model_file, tmp_path):
        model = ModEMModel.from_file(modem_model_file)
        out = tmp_path / "meshgen.inp"
        model.to_femtic_meshgen(out, init_res=500.0)
        assert out.exists()
        text = out.read_text()
        assert "INITIAL_RESISTIVITY" in text
        assert "END" in text


class TestPyvistaExports:
    """Optional pyvista-backed exports."""

    def test_to_vtk(self, modem_model_file, tmp_path):
        pytest.importorskip("pyvista")
        model = ModEMModel.from_file(modem_model_file)
        out = tmp_path / "model.vtk"
        model.to_vtk(out)
        assert out.exists()
