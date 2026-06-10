"""Shared base for FEMTIC mesh-input builders.

Module holds small :class:`FemticMesh` base class common to the
hexahedral and tetrahedral mesh builders. The builders themselves live in
their own modules.

* :mod:`hexmesh` — :class:`~hexmesh.DeformableHexMesh`
    (``makeDHexaMesh`` inputs)
* :mod:`tetramesh` — :class:`~tetramesh.TetraMesh`
    (``makeTetraMesh`` pipeline inputs)

@author: oaazeved

"""

from __future__ import annotations

import pathlib
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

PathLike = Union[str, pathlib.Path]

# Skin-depth constant: depth [m] ~= 500 * sqrt(rho[ohm-m] * T[s]).
_SKIN_DEPTH_CONST = 500.0

# Required columns in the mtpy-v2 long DataFrame.
_REQUIRED_COLS = ("station", "east", "north")


class FemticMesh:
    """Common base for FEMTIC mesh-input builders.

    :param mt_df: mtpy-v2 long DataFrame with at least ``station``,
        ``east``, ``north`` columns in meters (``period`` in seconds is
        required for skin-depth-based depth sizing).
    :type mt_df: pandas.DataFrame
    :param start_res: Assumed background resistivity (ohm-m) used for
        skin-depth sizing, defaults to ``100.0``.
    :type start_res: float, optional
    """

    def __init__(self, mt_df: pd.DataFrame, *, start_res: float = 100.0):
        missing = [c for c in _REQUIRED_COLS if c not in mt_df.columns]
        if missing:
            raise KeyError(
                f"mt_df is missing required column(s): {missing}. "
                f"Pass an mtpy-v2 dataframe (mt_data.to_dataframe())."
            )
        self.mt_df = mt_df.copy()
        self.start_res = float(start_res)
        self.logger = logger

    # alternative constructors 

    @classmethod
    def from_mt_dataframe(cls, mt_df: pd.DataFrame, **kwargs) -> "FemticMesh":
        """Build from an mtpy-v2 long DataFrame (alias for the constructor).

        :param mt_df: mtpy-v2 long DataFrame.
        :type mt_df: pandas.DataFrame
        :return: A new instance of the calling class.
        :rtype: FemticMesh
        """
        return cls(mt_df, **kwargs)

    @classmethod
    def from_mt_data(cls, mt_data, **kwargs) -> "FemticMesh":
        """Build from an mtpy :class:`mtpy.MTData` collection.

        Calls ``mt_data.to_dataframe()`` and forwards to the constructor.

        :param mt_data: An mtpy MTData instance.
        :return: A new instance of the calling class.
        :rtype: FemticMesh
        """
        return cls(mt_data.to_dataframe(), **kwargs)

    # shared accessors 

    @property
    def stations(self) -> np.ndarray:
        """Unique station identifiers, in first-seen order."""
        return self.mt_df["station"].unique()

    @property
    def n_stations(self) -> int:
        """Number of unique stations."""
        return int(self.mt_df["station"].nunique())

    @property
    def periods(self) -> np.ndarray:
        """Sorted unique periods (s), or an empty array if absent."""
        if "period" not in self.mt_df.columns:
            return np.array([])
        return np.sort(self.mt_df["period"].unique())

    @property
    def n_periods(self) -> int:
        """Number of unique periods."""
        return int(len(self.periods))

    @property
    def station_coords_m(self) -> pd.DataFrame:
        """Per-station ``station``/``east``/``north`` table (meters)."""
        return (self.mt_df[["station", "east", "north"]]
                .drop_duplicates(subset=["station"])
                .reset_index(drop=True))

    # shared geometry 

    def skin_depth_km(self, which: str = "max",
                        res: Optional[float] = None) -> float:
        """Skin depth (km) at the shortest or longest period.

        :param which: ``"max"`` for the longest period (deepest) or
            ``"min"`` for the shortest period (shallowest), defaults to
            ``"max"``.
        :type which: str, optional
        :param res: Resistivity (ohm-m) to use; defaults to
            :attr:`start_res`.
        :type res: float, optional
        :return: Skin depth in km.
        :rtype: float
        :raises ValueError: If ``which`` is not ``"max"`` or ``"min"``.
        """
        if "period" not in self.mt_df.columns:
            raise KeyError("mt_df has no 'period' column for skin-depth sizing.")
        rho = self.start_res if res is None else float(res)
        if which == "max":
            period = float(np.max(self.mt_df["period"]))
        elif which == "min":
            period = float(np.min(self.mt_df["period"]))
        else:
            raise ValueError("which must be 'max' or 'min'.")
        return _SKIN_DEPTH_CONST * np.sqrt(rho * period) / 1000.0

    def write_inputs(self, out_dir: PathLike):
        """Write every mesh-input file for this mesh type into ``out_dir``.

        Implemented by subclasses.

        :param out_dir: Destination directory (created if missing).
        :type out_dir: str or pathlib.Path
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n_stations={self.n_stations}, "
                f"n_periods={self.n_periods}, start_res={self.start_res})")
