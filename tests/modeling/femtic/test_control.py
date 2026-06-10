# -*- coding: utf-8 -*-
"""Tests for :mod:`mtpy.modeling.femtic.control`.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths
- All file I/O uses per-test tmp_path
"""

from __future__ import annotations

import pytest

from mtpy.modeling.femtic.control import write_inv_control, write_sbatch


class TestWriteSbatch:
    def test_writes_named_file(self, tmp_path):
        write_sbatch(
            tmp_path,
            mailuser="me@example.com",
            account="acct",
            qos="normal",
            runprogram="makeDHexaMesh",
            filename="meshjob",
        )
        out = tmp_path / "meshjob.sbatch"
        assert out.exists()

    def test_script_contains_slurm_directives(self, tmp_path):
        write_sbatch(
            tmp_path,
            mailuser="me@example.com",
            account="acct",
            qos="normal",
            runprogram="femtic",
            filename="invjob",
            jobname="inv_run",
        )
        text = (tmp_path / "invjob.sbatch").read_text()
        assert text.startswith("#!")
        assert "#SBATCH" in text
        assert "inv_run" in text


class TestWriteInvControl:
    def test_writes_control_dat(self, tmp_path):
        write_inv_control(filepath=tmp_path)
        assert (tmp_path / "control.dat").exists()

    def test_control_dat_not_empty(self, tmp_path):
        write_inv_control(filepath=tmp_path)
        assert (tmp_path / "control.dat").stat().st_size > 0
