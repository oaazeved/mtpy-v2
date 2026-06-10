# -*- coding: utf-8 -*-
"""
FEMTIC
==========

Read/write inputs and parse outputs for the FEMTIC 3-D magnetotelluric
forward-modeling and inversion code, in the style of the other
:mod:`mtpy.modeling` submodules.

The submodules are organised by concern:

* :mod:`~mtpy.modeling.femtic.mesh` -- shared :class:`FemticMesh` base.
* :mod:`~mtpy.modeling.femtic.hexmesh` -- :class:`DeformableHexMesh`
  (``makeDHexaMesh`` inputs).
* :mod:`~mtpy.modeling.femtic.tetramesh` -- :class:`TetraMesh`
  (``makeTetraMesh`` pipeline inputs).
* :mod:`~mtpy.modeling.femtic.responses` -- ``observe.dat`` I/O,
  inversion responses, convergence, and RMS analysis.
* :mod:`~mtpy.modeling.femtic.modem` -- ModEM <-> FEMTIC data/model
  bridges.
* :mod:`~mtpy.modeling.femtic.control` -- control-file and SLURM
  job-script writers.
* :mod:`~mtpy.modeling.femtic.remote` -- Slurm-cluster SSH/SFTP helpers.
"""

from .mesh import FemticMesh
from .hexmesh import DeformableHexMesh
from .tetramesh import TetraMesh
from .responses import (
    FemticData,
    FemticResponses,
    FemticConvergence,
    RMSAnalysis,
    FemticInversion,
    write_mtdata,
    check_error_floor,
    apply_error_floor,
    convert_units,
    conjugate,
    drop_nodata,
    prep_data,
)
from .modem import (
    ModEMModel,
    write_modem_to_femtic,
    write_femtic_to_modem,
    modem_data_to_femtic_obssite,
)
from .control import write_inv_control, write_sbatch
from .remote import (
    RemoteCluster,
    RemoteCommandResult,
    SqueueResult,
    parse_sbatch_job_id,
    parse_squeue_stdout,
    columns_from_squeue_format,
)


__all__ = [
    "FemticMesh",
    "DeformableHexMesh",
    "TetraMesh",
    "FemticData",
    "FemticResponses",
    "FemticConvergence",
    "RMSAnalysis",
    "FemticInversion",
    "write_mtdata",
    "check_error_floor",
    "apply_error_floor",
    "convert_units",
    "conjugate",
    "drop_nodata",
    "prep_data",
    "ModEMModel",
    "write_modem_to_femtic",
    "write_femtic_to_modem",
    "modem_data_to_femtic_obssite",
    "write_inv_control",
    "write_sbatch",
    "RemoteCluster",
    "RemoteCommandResult",
    "SqueueResult",
    "parse_sbatch_job_id",
    "parse_squeue_stdout",
    "columns_from_squeue_format",
]
