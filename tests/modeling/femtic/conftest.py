# -*- coding: utf-8 -*-
"""Shared fixtures for the :mod:`mtpy.modeling.femtic` test suite.

These fixtures are pytest-xdist safe: each returns a freshly built object
so there is no shared mutable state between tests, and all file I/O in the
individual modules uses the per-test ``tmp_path`` fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def station_frame() -> pd.DataFrame:
    """A minimal mtpy-v2 long DataFrame (station/east/north/period).

    Three stations, two periods each, with coordinates in meters and
    periods in seconds. Sufficient to drive the mesh builders and the
    skin-depth sizing without any optional GIS dependencies.
    """
    return pd.DataFrame(
        {
            "station": ["a", "a", "b", "b", "c", "c"],
            "east": [0.0, 0.0, 1000.0, 1000.0, 2000.0, 2000.0],
            "north": [0.0, 0.0, 500.0, 500.0, 1000.0, 1000.0],
            "period": [0.01, 1.0, 0.01, 1.0, 0.01, 1.0],
        }
    )


@pytest.fixture
def impedance_frame() -> pd.DataFrame:
    """A minimal mtpy-v2 long DataFrame carrying impedance columns.

    Used by the unit-conversion / conjugation / error-floor helpers in
    :mod:`mtpy.modeling.femtic.responses`.
    """
    n = 2
    z = np.array([0.1 - 0.1j, 10 + 10j, -10 - 10j, -0.1 + 0.1j])
    return pd.DataFrame(
        {
            "station": ["a", "a"],
            "period": [0.01, 1.0],
            "east": [0.0, 0.0],
            "north": [0.0, 0.0],
            "z_xx": np.repeat(z[0], n),
            "z_xy": np.repeat(z[1], n),
            "z_yx": np.repeat(z[2], n),
            "z_yy": np.repeat(z[3], n),
            "z_xx_error": np.repeat(0.1, n),
            "z_xy_error": np.repeat(0.5, n),
            "z_yx_error": np.repeat(0.5, n),
            "z_yy_error": np.repeat(0.1, n),
            "t_zx": np.repeat(0.1 + 0.05j, n),
            "t_zy": np.repeat(-0.1 - 0.05j, n),
        }
    )


@pytest.fixture
def observe_dat(tmp_path):
    """Write a minimal two-station FEMTIC ``observe.dat`` (MT block).

    Returns the path to the file. Imaginary parts are written in the
    FEMTIC ``exp(-i ω t)`` convention so that
    :meth:`FemticData.from_file` flips them back to mtpy's ``+`` sign.
    """
    sd = "  ".join(["0.0100"] * 8)
    z = "  ".join(
        [
            "1.0000e-01", "-1.0000e-01",
            "1.0000e+01", "1.0000e+01",
            "-1.0000e+01", "-1.0000e+01",
            "-1.0000e-01", "1.0000e-01",
        ]
    )
    lines = ["MT  2"]
    for sta, (x, y) in enumerate([(0.0, 0.0), (0.5, 1.0)], start=1):
        lines.append(f"{sta}  {sta}  {x:.4f}  {y:.4f}")
        lines.append("2")
        lines.append(f"100.0  {z}  {sd}")
        lines.append(f"1.0  {z}  {sd}")
    path = tmp_path / "observe.dat"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def convergence_file(tmp_path):
    """Write a minimal ``femtic.cnv`` convergence file and return its path."""
    # _keep_last_retrial keeps only the highest Retrial# per Iter#, so the
    # Iter# 2 / Retrial# 0 row (2.8) is dropped in favour of Retrial# 1 (2.0).
    text = (
        "Iter# Retrial# RMS Damping Roughness\n"
        "0 0 5.0 1.0 100.0\n"
        "1 0 3.0 1.0 80.0\n"
        "2 0 2.8 1.0 70.0\n"
        "2 1 2.0 1.0 72.0\n"
    )
    path = tmp_path / "femtic.cnv"
    path.write_text(text)
    return path


@pytest.fixture
def modem_model_file(tmp_path):
    """Write a minimal 2x2x2 ModEM rectilinear model file; return its path."""
    text = (
        "# test model\n"
        "  2  2  2  0 LOGE\n"
        "     100.000     100.000\n"
        "     100.000     100.000\n"
        "      50.000      50.000\n"
        "\n"
        "  1.00000E+02  1.00000E+02\n"
        "  1.00000E+02  1.00000E+02\n"
        "\n"
        "  2.00000E+02  2.00000E+02\n"
        "  2.00000E+02  2.00000E+02\n"
        "0.0 0.0 0.0\n"
        "0.0"
    )
    path = tmp_path / "model.rho"
    path.write_text(text)
    return path
