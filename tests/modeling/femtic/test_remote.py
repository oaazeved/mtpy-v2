# -*- coding: utf-8 -*-
"""Tests for :mod:`mtpy.modeling.femtic.remote`.

This suite is designed to be pytest-xdist safe:
- No global mutable state
- No shared file paths

Only the pure stdout parsers are exercised here, since they need no SSH
connection and no paramiko. :class:`RemoteCluster` construction is checked
only for its paramiko guard.
"""

from __future__ import annotations

import pytest

from mtpy.modeling.femtic.remote import (
    RemoteCluster,
    columns_from_squeue_format,
    parse_sbatch_job_id,
    parse_squeue_stdout,
)


class TestParseSbatchJobId:
    def test_extracts_job_id(self):
        assert parse_sbatch_job_id("Submitted batch job 123456") == 123456

    def test_returns_none_when_absent(self):
        assert parse_sbatch_job_id("no job here") is None

    def test_handles_empty_input(self):
        assert parse_sbatch_job_id("") is None


class TestColumnsFromSqueueFormat:
    def test_maps_known_tokens(self):
        cols = columns_from_squeue_format("%.18i %.9P %.20j %.8u %.2t")
        assert cols == ["JOBID", "PARTITION", "NAME", "USER", "STATE"]

    def test_unknown_token_upper_cased(self):
        # An unmapped key falls back to its upper-cased letter.
        cols = columns_from_squeue_format("%i %z")
        assert cols[0] == "JOBID"
        assert cols[1] == "Z"


class TestParseSqueueStdout:
    def test_default_generic_columns(self):
        rows = parse_squeue_stdout("7 run RUNNING\n8 x PENDING")
        assert rows[0] == {"col0": "7", "col1": "run", "col2": "RUNNING"}
        assert len(rows) == 2

    def test_named_columns_with_delimiter(self):
        rows = parse_squeue_stdout(
            "7|run|RUNNING", delimiter="|", columns=["JOBID", "NAME", "STATE"]
        )
        assert rows == [{"JOBID": "7", "NAME": "run", "STATE": "RUNNING"}]

    def test_skips_blank_lines(self):
        rows = parse_squeue_stdout("7 run RUNNING\n\n8 x PENDING")
        assert len(rows) == 2

    def test_empty_input_returns_empty_list(self):
        assert parse_squeue_stdout("") == []

    def test_field_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="expected 3 fields"):
            parse_squeue_stdout("7 run", columns=["JOBID", "NAME", "STATE"])


class TestRemoteClusterGuard:
    def test_construction_without_paramiko(self):
        """Without paramiko, constructing a RemoteCluster must error clearly."""
        paramiko = pytest.importorskip  # noqa: F841
        try:
            import paramiko as _p  # noqa: F401

            has_paramiko = True
        except ImportError:
            has_paramiko = False

        if has_paramiko:
            pytest.skip("paramiko installed; guard path not exercised")

        with pytest.raises(ImportError, match="paramiko"):
            RemoteCluster("host", "user", "pw", connect=False)
