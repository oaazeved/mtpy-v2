"""FEMTIC data I/O, responses, and analysis.

Module provides object-oriented access to FEMTIC inversion responses
and the ``observe.dat`` writing functionality used to set up an inversion.

covers two directions:

* :func:`write_mtdata` ingests an **mtpy-v2 long DataFrame**
  (``mt_data.to_dataframe()`` — columns ``station``, ``period``,
  ``east``, ``north``, ``z_xx``/``z_xx_error``/…, and optionally
  ``res_*``/``phase_*``/``t_z*``) and writes a FEMTIC ``observe.dat``
  with ``MT`` (impedance), ``APP_RES_AND_PHS``, and/or ``VTF`` blocks.
  Applies unit conversion (mV/km/nT -> ohm), the FEMTIC phase 
  conjugation, and an error floor before writing. 
* :meth:`FemticData.to_file` writes ``observe.dat`` from the wide scheme
  this module parses (e.g. after :meth:`FemticData.from_modem`).


Conventions

FEMTIC stores impedances in **ohms** under the ``exp(-i ω t)``
convention; mtpy and ModEM use **[mV/km]/[nT]** under ``exp(+i ω t)``.
Each class should track its current ``units`` as ``"ohm"`` or 
``"mV/km/nT"`` and exposes a :meth:`to_units` method to convert lazily.

@author: oaazeved

"""

from __future__ import annotations

import copy
import pathlib
import re
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from mtpy import MT
from mtpy.core.mt_data import MTData
from mtpy.core.mt_collection import MTCollection


# FEMTIC sentinel for missing data in observed and SD columns
NODATA_VAL: float = -9.999e-05

# Multiplier converting impedance in ohm to [mV/km]/[nT]
OHM_TO_MVKMNT: float = 10000.0 / (4.0 * np.pi)

# Impedance modes in FEMTIC order
MODES = ("Zxx", "Zxy", "Zyx", "Zyy")

# Mode -> matplotlib color used by the built-in plotters
COMPONENT_COLORS = {"Zxx": "tab:pink", "Zxy": "tab:red",
                    "Zyx": "tab:blue", "Zyy": "tab:green"}

# Supported impedance unit names
_UNIT_OHM = "ohm"
_UNIT_MODEM = "mV/km/nT"
_VALID_UNITS = (_UNIT_OHM, _UNIT_MODEM)


PathLike = Union[str, pathlib.Path]


# Private helpers
def _remove_trailing_comma(filename: PathLike) -> None:
    """Strip stray ``,\\n`` line endings that FEMTIC sometimes writes."""
    with open(filename, "r") as f:
        lines = f.readlines()
    new = [re.sub(r",\n", "\n", line, flags=re.MULTILINE | re.IGNORECASE) for line in lines]
    if new != lines:
        with open(filename, "w") as f:
            f.writelines(new)


def _format_code(value) -> str:
    """Render a station identifier as a string, stripping ``.0`` from floats."""
    try:
        f = float(value)
        if not np.isnan(f) and f == int(f):
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value)


def _rms(values) -> float:
    """Return ``sqrt(mean(values**2))`` ignoring NaN."""
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(a ** 2)))


def _check_units(units: str) -> str:
    """Validate a units string and return it unchanged."""
    if units not in _VALID_UNITS:
        raise ValueError(
            f"Invalid units {units!r}; must be one of {_VALID_UNITS}."
        )
    return units


def _resolve_station_col(df: pd.DataFrame,
                         requested: Optional[str] = None) -> str:
    """Pick a station-id column, falling back through common names."""
    if requested is not None:
        if requested not in df.columns:
            raise KeyError(f"Column {requested!r} not in DataFrame.")
        return requested
    for candidate in ("StaID", "Station", "Code"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No station-id column found; pass station_col explicitly.")


def _coerce_to_obs_dataframe(source) -> pd.DataFrame:
    """Normalize an error-floor-check input to a wide observation DataFrame.

    Accepts a DataFrame, a :class:`FemticData`, a :class:`FemticResponses`,
    a path to an ``observe.dat`` file, or a path to a single
    ``result_<freq>_iter<iter>.csv`` file. Returns a DataFrame in
    FEMTIC's wide schema, with NaN substituted for the nodata sentinel.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, FemticData):
        df = source.mask_nodata().to_dataframe()
    elif isinstance(source, FemticResponses):
        df = source.to_dataframe()
    else:
        path = pathlib.Path(source)
        if path.suffix.lower() == ".csv":
            _remove_trailing_comma(path)
            df = pd.read_csv(path, header=1, usecols=range(0, 34))
            df.columns = df.columns.str.strip()
        else:
            df = FemticData.from_file(path).mask_nodata().to_dataframe()
    # Replace any remaining no-data sentinel with NaN in obs/SD columns
    for col in df.columns:
        if ("_SD" in col or "_Obs" in col
                or (col.startswith(("Re(", "Im(")) and "_" not in col)):
            if pd.api.types.is_numeric_dtype(df[col]):
                mask = np.isclose(df[col].to_numpy(dtype=float),
                                  NODATA_VAL, atol=1e-9)
                df.loc[mask, col] = np.nan
    return df


def check_error_floor(source,
                      floor_pct: Optional[float] = None,
                      floor_type: str = "off_diagonal_geometric",
                      tol: float = 1e-3,
                      verbose: bool = False) -> dict:
    """Check whether a FEMTIC observation set has an error floor applied.

    An *error floor* is a minimum value imposed on the impedance
    standard deviations so that points with very small reported error
    are not over-weighted by the inversion. After a floor of fraction
    ``p`` has been applied, every standard deviation should satisfy
    ``σ ≥ p · |reference|``, where the reference depends on the
    convention (geometric mean of off-diagonal magnitudes is the most
    common in MT processing).

    :param source: One of:

        - :class:`pandas.DataFrame` in FEMTIC's wide schema
        - :class:`FemticData` instance
        - :class:`FemticResponses` instance
        - path (``str`` or :class:`pathlib.Path`) to an ``observe.dat``
        - path to a single ``result_<freq>_iter<iter>.csv``

    :param floor_pct: Expected floor as a fraction (e.g. ``0.05`` for
        5%). If None (default), the empirically-observed floor is
        inferred from the data and reported but no pass/fail check is
        performed.
    :type floor_pct: float, optional
    :param floor_type: Which floor convention to check. One of

        - ``"off_diagonal_geometric"``: reference is
          ``sqrt(|Zxy| · |Zyx|)``; only Zxy and Zyx errors checked.
        - ``"off_diagonal_each"``: reference is ``|Zxy|`` for the Zxy
          error and ``|Zyx|`` for the Zyx error.
        - ``"all_each"``: reference is ``|Z_ij|`` for each of the four
          modes' errors.

        Defaults to ``"off_diagonal_geometric"``.
    :type floor_type: str, optional
    :param tol: Relative tolerance for the floor check, defaults to
        ``1e-3`` (i.e. errors must be ≥ ``floor_pct · (1 - tol) · ref``).
    :type tol: float, optional
    :param verbose: If True, print the per-mode summary table. Defaults
        to False.
    :type verbose: bool, optional

    :return: A dict with keys

        - ``"passes"`` (bool, only meaningful when ``floor_pct`` is set)
        - ``"floor_pct_checked"`` (float or None)
        - ``"floor_pct_inferred"`` (float, the minimum observed ratio)
        - ``"floor_type"`` (str)
        - ``"n_valid"`` (int, total valid (station, period, mode) points)
        - ``"n_below_floor"`` (int)
        - ``"per_mode"`` (dict): per-mode stats with keys
          ``n_valid``, ``min_rel_err``, ``median_rel_err``,
          ``n_below_floor``.

    :rtype: dict

    :raises ValueError: If ``floor_type`` is not a recognized option.
    """
    valid_types = ("off_diagonal_geometric", "off_diagonal_each", "all_each")
    if floor_type not in valid_types:
        raise ValueError(f"floor_type must be one of {valid_types}, "
                         f"got {floor_type!r}.")

    df = _coerce_to_obs_dataframe(source)

    use_obs_suffix = "Re(Zxx)_Obs" in df.columns
    def re_col(m): return f"Re({m})_Obs" if use_obs_suffix else f"Re({m})"
    def im_col(m): return f"Im({m})_Obs" if use_obs_suffix else f"Im({m})"
    def sd_col(m): return f"Re({m})_SD"

    # Compute |Z| for each mode
    z_mag = {}
    for mode in MODES:
        if re_col(mode) not in df.columns or im_col(mode) not in df.columns:
            z_mag[mode] = None
            continue
        zr = df[re_col(mode)].to_numpy(dtype=float)
        zi = df[im_col(mode)].to_numpy(dtype=float)
        z_mag[mode] = np.abs(zr + 1j * zi)

    # Modes to check
    if floor_type == "all_each":
        modes_checked = list(MODES)
    else:
        modes_checked = ["Zxy", "Zyx"]

    # Reference computation per mode
    if floor_type == "off_diagonal_geometric":
        if z_mag["Zxy"] is None or z_mag["Zyx"] is None:
            raise ValueError("Off-diagonal data required for geometric floor.")
        ref_common = np.sqrt(z_mag["Zxy"] * z_mag["Zyx"])

    per_mode = {}
    all_rels = []
    for mode in modes_checked:
        if z_mag[mode] is None or sd_col(mode) not in df.columns:
            continue
        sd = df[sd_col(mode)].to_numpy(dtype=float)
        ref = ref_common if floor_type == "off_diagonal_geometric" else z_mag[mode]
        valid = ((sd > 0) & (ref > 0)
                 & np.isfinite(sd) & np.isfinite(ref))
        rel = np.full_like(sd, np.nan, dtype=float)
        rel[valid] = sd[valid] / ref[valid]
        rel_v = rel[valid]
        if rel_v.size == 0:
            per_mode[mode] = {"n_valid": 0, "min_rel_err": float("nan"),
                              "median_rel_err": float("nan"),
                              "n_below_floor": 0}
            continue
        n_below_mode = (int(np.sum(rel_v < floor_pct * (1.0 - tol)))
                        if floor_pct is not None else 0)
        per_mode[mode] = {
            "n_valid": int(rel_v.size),
            "min_rel_err": float(rel_v.min()),
            "median_rel_err": float(np.median(rel_v)),
            "n_below_floor": n_below_mode,
        }
        all_rels.append(rel_v)

    if not all_rels:
        result = {
            "passes": False, "floor_pct_checked": floor_pct,
            "floor_pct_inferred": float("nan"), "floor_type": floor_type,
            "n_valid": 0, "n_below_floor": 0, "per_mode": per_mode,
        }
    else:
        all_rels = np.concatenate(all_rels)
        inferred = float(all_rels.min())
        if floor_pct is None:
            passes = True
            n_below = 0
        else:
            n_below = int(np.sum(all_rels < floor_pct * (1.0 - tol)))
            passes = n_below == 0
        result = {
            "passes": passes,
            "floor_pct_checked": floor_pct,
            "floor_pct_inferred": inferred,
            "floor_type": floor_type,
            "n_valid": int(all_rels.size),
            "n_below_floor": n_below,
            "per_mode": per_mode,
        }

    if verbose:
        logger.info(_format_floor_result(result))
    return result


def _format_floor_result(result: dict) -> str:
    """Render a :func:`check_error_floor` result as a human-readable summary."""
    lines = []
    lines.append(f"Error-floor check ({result['floor_type']})")
    if result["floor_pct_checked"] is not None:
        verdict = "PASS" if result["passes"] else "FAIL"
        lines.append(f"  Target floor:    {result['floor_pct_checked']:.4%}  ->  {verdict}")
        lines.append(f"  Below target:    {result['n_below_floor']} / "
                     f"{result['n_valid']} points")
    lines.append(f"  Inferred floor:  {result['floor_pct_inferred']:.4%} "
                 f"(minimum observed σ/|ref|)")
    lines.append("  Per-mode:")
    lines.append(f"    {'mode':6}{'n':>8}{'min %':>14}{'median %':>14}"
                 f"{'n_below':>10}")
    for mode, pm in result["per_mode"].items():
        lines.append(
            f"    {mode:6}{pm['n_valid']:>8d}"
            f"{pm['min_rel_err']*100:>13.4f}%"
            f"{pm['median_rel_err']*100:>13.4f}%"
            f"{pm['n_below_floor']:>10d}"
        )
    return "\n".join(lines)


def _match_stations_by_position(femtic_coords: pd.DataFrame,
                                external: pd.DataFrame,
                                code_col: str = "Code",
                                ex_x_col: str = "X",
                                ex_y_col: str = "Y",
                                check_swap: bool = True) -> dict:
    """Return a ``{femtic_id: external_name}`` map by spatial nearest-neighbor.

    Both networks are recentered on their centroid before matching, so
    arbitrary translations (different model origins, different
    coordinate frames) are accepted. If ``check_swap`` is True, the
    function also tries swapping the external X/Y axes and keeps
    whichever gives a smaller total assignment distance; this should 
    catch the case where one source uses (east, north) and the 
    other uses (north, east), but this will fail in places where 
    the domain/range of the coordinates is comparable.

    :param femtic_coords: DataFrame with columns ``Station``, ``X``,
        ``Y`` (FEMTIC convention: X=east, Y=north, in km).
    :type femtic_coords: pandas.DataFrame
    :param external: DataFrame with columns ``code_col``, ``ex_x_col``,
        ``ex_y_col``.
    :type external: pandas.DataFrame
    :param code_col: Name column in ``external``, defaults to ``"Code"``.
    :type code_col: str, optional
    :param ex_x_col: X column in ``external``, defaults to ``"X"``.
    :type ex_x_col: str, optional
    :param ex_y_col: Y column in ``external``, defaults to ``"Y"``.
    :type ex_y_col: str, optional
    :param check_swap: If True, also try swapping ``X`` <-> ``Y`` in the
        external coords and keep the better match, defaults to True.
    :type check_swap: bool, optional

    :return: ``{int_femtic_station_id: external_name_str}``.
    :rtype: dict
    """
    f_ids = femtic_coords["Station"].to_numpy()
    f_xy = femtic_coords[["X", "Y"]].to_numpy(dtype=float)
    ex_names = external[code_col].astype(str).to_numpy()
    ex_xy = external[[ex_x_col, ex_y_col]].to_numpy(dtype=float)

    f_c = f_xy - f_xy.mean(axis=0)
    candidates = [ex_xy - ex_xy.mean(axis=0)]
    if check_swap:
        sw = ex_xy[:, [1, 0]]
        candidates.append(sw - sw.mean(axis=0))

    best_dict, best_cost = None, np.inf
    for cand in candidates:
        # Scale so both have unit RMS distance from centroid (handles km vs m)
        f_scale = np.sqrt(np.mean(np.sum(f_c ** 2, axis=1)))
        e_scale = np.sqrt(np.mean(np.sum(cand ** 2, axis=1)))
        if f_scale == 0 or e_scale == 0:
            continue
        f_n = f_c / f_scale
        e_n = cand / e_scale
        # Pairwise distance: (n_f, n_e)
        d = np.linalg.norm(f_n[:, None, :] - e_n[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        cost = float(d[np.arange(len(f_ids)), nearest].sum())
        if cost < best_cost:
            best_cost = cost
            best_dict = {int(f_ids[i]): str(ex_names[nearest[i]])
                         for i in range(len(f_ids))}
    return best_dict


def _station_title(sta, iteration, name_dict: Optional[dict],
                   rms: Optional[float] = None) -> str:
    """Build a plot title like ``"NIC01 (Sta 1, iter 11) — RMS: 11.41"``.

    Falls back to ``"Station {sta} (iter {iteration})"`` when the
    station has no entry in ``name_dict``. The RMS suffix is appended
    when ``rms`` is a finite number.
    """
    sta_key = None
    try:
        sta_key = int(sta)
    except (TypeError, ValueError):
        pass
    name = None
    if name_dict and sta_key is not None and sta_key in name_dict:
        name = name_dict[sta_key]
    elif name_dict and str(sta) in name_dict:
        name = name_dict[str(sta)]
    title = (f"{name} (Sta {sta}, iter {iteration})"
             if name else f"Station {sta} (iter {iteration})")
    if rms is not None and np.isfinite(rms):
        title += f" — RMS: {rms:.2f}"
    return title


def _station_filename_suffix(sta, name_dict: Optional[dict]) -> str:
    """Return a filename-safe ``"sta{N}"`` or ``"sta{N}_NAME"`` suffix.

    Use as ``f"response_{kind}_{_station_filename_suffix(sta, name_dict)}.png"``.
    """
    sta_key = None
    try:
        sta_key = int(sta)
    except (TypeError, ValueError):
        pass
    name = None
    if name_dict and sta_key is not None and sta_key in name_dict:
        name = name_dict[sta_key]
    elif name_dict and str(sta) in name_dict:
        name = name_dict[str(sta)]
    if name:
        # Strip path separators / whitespace from the name just in case
        safe = re.sub(r"[\s/\\]+", "_", str(name))
        return f"sta{sta}_{safe}"
    return f"sta{sta}"


def _read_h5_station_dataframe(filepath: PathLike) -> pd.DataFrame:
    """Extract station name + (east, north) from an mtpy-v2 HDF5 file.

    :param filepath: Path to the ``.h5`` collection.
    :type filepath: str or pathlib.Path

    :return: DataFrame with columns ``Code``, ``X`` (east, in km),
        ``Y`` (north, in km) so the result can be used as the
        ``external`` argument to :func:`_match_stations_by_position`.
    :rtype: pandas.DataFrame

    :raises ImportError: If mtpy is not installed.
    """
    # users on slightly different mtpy versions may need to use attach_names() 
    # with an explicit dict instead.
    mc = MTCollection()
    mc.open_collection(filename=str(filepath))
    df = getattr(mc, "dataframe", None)
    code_col = "station" if "station" in df.columns else "Station"
    if "east" in df.columns and "north" in df.columns:
        east, north = df["east"].to_numpy(), df["north"].to_numpy()
    else:
        # Fall back to lat/lon if east/north not present
        east = df["longitude"].to_numpy()
        north = df["latitude"].to_numpy()
    coords = (pd.DataFrame({"Code": df[code_col].astype(str).to_numpy(),
                            "X": east / 1000.0, "Y": north / 1000.0})
              .drop_duplicates(subset=["Code"]).reset_index(drop=True))
    return coords


 # FemticData; wraps observe.dat
 
class FemticData:
    """Container for a FEMTIC ``observe.dat`` (observed impedance data).

    The class wraps a single wide-format DataFrame with one row per
    (station, frequency). Use :meth:`from_file` to construct from a
    FEMTIC observation file, :meth:`from_modem` to construct from a
    ModEM ``.data`` file, or pass a pre-built DataFrame to the
    constructor directly.

    :param dataframe: A DataFrame in the wide FEMTIC schema. If None
        (default), an empty container is created.
    :type dataframe: pandas.DataFrame, optional
    :param units: Unit of the impedance and SD columns. Either
        ``"ohm"`` (FEMTIC native) or ``"mV/km/nT"`` (ModEM/MTpy
        native), defaults to ``"ohm"``.
    :type units: str, optional
    :param phase_convention: ``"-"`` for FEMTIC's ``exp(-i ω t)`` or
        ``"+"`` for ModEM's ``exp(+i ω t)``. Affects the sign of
        ``Im(...)`` columns. Defaults to ``"+"`` because the file
        readers flip the sign on input.
    :type phase_convention: str, optional
    """

    # Wide-schema columns produced by :meth:`from_file`.
    _COLUMNS = (
        "Station", "StaID2", "owner", "ftype", "X", "Y",
        "Freq[Hz]", "Period[s]",
        "Re(Zxx)", "Im(Zxx)", "Re(Zxy)", "Im(Zxy)",
        "Re(Zyx)", "Im(Zyx)", "Re(Zyy)", "Im(Zyy)",
        "Re(Zxx)_SD", "Im(Zxx)_SD", "Re(Zxy)_SD", "Im(Zxy)_SD",
        "Re(Zyx)_SD", "Im(Zyx)_SD", "Re(Zyy)_SD", "Im(Zyy)_SD",
    )

    def __init__(self,
                 dataframe: Optional[pd.DataFrame] = None,
                 units: str = _UNIT_OHM,
                 phase_convention: str = "+"):
        self._df = dataframe.copy() if dataframe is not None else pd.DataFrame()
        self._units = _check_units(units)
        if phase_convention not in ("+", "-"):
            raise ValueError("phase_convention must be '+' or '-'.")
        self._phase = phase_convention
        self.logger = logger

    # alternative constructors -

    @classmethod
    def from_file(cls, filepath: PathLike, units: str = _UNIT_OHM) -> "FemticData":
        """Parse a FEMTIC ``observe.dat`` into a :class:`FemticData`.

        Imaginary signs are flipped on read so the returned object is
        in the ``exp(+i ω t)`` convention used by mtpy and ModEM.

        :param filepath: Path to the file.
        :type filepath: str or pathlib.Path
        :param units: Unit to interpret the values in. ``"ohm"`` (the
            FEMTIC native unit) preserves the values as written;
            ``"mV/km/nT"`` multiplies them by :data:`OHM_TO_MVKMNT`.
            Defaults to ``"ohm"``.
        :type units: str, optional

        :return: A :class:`FemticData` instance.
        :rtype: FemticData
        """
        records = []
        with open(filepath, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                data_type = parts[0]
                n_stations = int(parts[1])
                for _ in range(n_stations):
                    sp = f.readline().split()
                    sta_id = int(sp[0])
                    sta_2nd_id = int(sp[1])
                    optional = sp[2:-2]
                    owner = optional[0] if len(optional) >= 1 else None
                    ftype = optional[1] if len(optional) >= 2 else None
                    x = float(sp[-2])
                    y = float(sp[-1])
                    n_freqs = int(f.readline().strip())
                    for _ in range(n_freqs):
                        fp = f.readline().split()
                        freq = float(fp[0])
                        v = list(map(float, fp[1:]))
                        records.append({
                            "datatype":   data_type,
                            "Station":    sta_id,
                            "StaID2":     sta_2nd_id,
                            "owner":      owner,
                            "ftype":      ftype,
                            "X":          x,
                            "Y":          y,
                            "Freq[Hz]":   freq,
                            "Period[s]":  1.0 / freq,
                            "Re(Zxx)":    v[0],  "Im(Zxx)":    -v[1],
                            "Re(Zxy)":    v[2],  "Im(Zxy)":    -v[3],
                            "Re(Zyx)":    v[4],  "Im(Zyx)":    -v[5],
                            "Re(Zyy)":    v[6],  "Im(Zyy)":    -v[7],
                            "Re(Zxx)_SD": v[8],  "Im(Zxx)_SD": v[9],
                            "Re(Zxy)_SD": v[10], "Im(Zxy)_SD": v[11],
                            "Re(Zyx)_SD": v[12], "Im(Zyx)_SD": v[13],
                            "Re(Zyy)_SD": v[14], "Im(Zyy)_SD": v[15],
                        })
        df = pd.DataFrame(records)
        obj = cls(df, units=_UNIT_OHM, phase_convention="+")
        if units == _UNIT_MODEM:
            obj = obj.to_units(_UNIT_MODEM)
        return obj

    @classmethod
    def from_modem(cls, filepath: PathLike) -> "FemticData":
        """Parse a ModEM ``.data`` file (Full_Impedance block).

        The result is in ``"mV/km/nT"`` units and the ``exp(+i ω t)``
        convention. Stations whose coordinates are missing are dropped.

        :param filepath: Path to the ModEM data file.
        :type filepath: str or pathlib.Path

        :return: A :class:`FemticData` instance.
        :rtype: FemticData
        """
        long_rows = []
        in_impedance = False
        with open(filepath, "r") as f:
            for line in f:
                ln = line.strip()
                if not ln or ln.startswith("#"):
                    continue
                if ln.startswith(">"):
                    if "impedance" in ln.lower():
                        in_impedance = True
                    elif "vertical" in ln.lower():
                        in_impedance = False
                    continue
                if not in_impedance:
                    continue
                parts = ln.split()
                if len(parts) < 11:
                    continue
                try:
                    long_rows.append({
                        "Period[s]":  float(parts[0]),
                        "Code":       parts[1],
                        "Lat":        float(parts[2]),
                        "Lon":        float(parts[3]),
                        "X":          float(parts[4]) / 1000.0,  # m -> km
                        "Y":          float(parts[5]) / 1000.0,
                        "Component":  parts[7].upper(),
                        "Real":       float(parts[8]),
                        "Imag":       float(parts[9]),
                        "Error":      float(parts[10]),
                    })
                except ValueError:
                    continue

        if not long_rows:
            return cls(units=_UNIT_MODEM)

        long_df = pd.DataFrame(long_rows)
        # Pivot to wide: one row per (station, period), columns per component.
        codes = sorted(long_df["Code"].unique())
        code_to_id = {c: i + 1 for i, c in enumerate(codes)}
        records = []
        for (code, period), grp in long_df.groupby(["Code", "Period[s]"]):
            row = {
                "Station":   code_to_id[code],
                "StaID2":    code_to_id[code],
                "owner":     code,         # carry the station name through
                "ftype":     None,
                "X":         grp["X"].iloc[0],   # km
                "Y":         grp["Y"].iloc[0],
                "Freq[Hz]":  1.0 / period,
                "Period[s]": period,
            }
            # Note: ModEM/X is north, Y is east — swap for FEMTIC convention
            # where X is east, Y is north (as used in observe.dat).
            row["X"], row["Y"] = grp["Y"].iloc[0], grp["X"].iloc[0]
            for mode in MODES:
                m = grp[grp["Component"] == mode.upper()]
                if len(m) == 1:
                    row[f"Re({mode})"]    = float(m["Real"].iloc[0])
                    row[f"Im({mode})"]    = float(m["Imag"].iloc[0])
                    row[f"Re({mode})_SD"] = float(m["Error"].iloc[0])
                    row[f"Im({mode})_SD"] = float(m["Error"].iloc[0])
                else:
                    row[f"Re({mode})"]    = np.nan
                    row[f"Im({mode})"]    = np.nan
                    row[f"Re({mode})_SD"] = np.nan
                    row[f"Im({mode})_SD"] = np.nan
            records.append(row)
        df = pd.DataFrame(records).sort_values(["Station", "Period[s]"]).reset_index(drop=True)
        return cls(df, units=_UNIT_MODEM, phase_convention="+")

    @classmethod
    def from_mt_data(cls, mt_data) -> "FemticData":
        """Build from an mtpy :class:`mtpy.MTData` collection.

        :param mt_data: An mtpy MTData instance.
        :type mt_data: mtpy.MTData

        :return: A :class:`FemticData` with one row per (station,
            period) of the input collection, in ``"mV/km/nT"`` units.
        :rtype: FemticData

        :raises ImportError: If mtpy is not installed.
        """

        records = []
        for idx, mt in enumerate(mt_data, start=1):
            z = mt.Z
            for i_per, period in enumerate(z.period):
                z_mat = z.z[i_per]
                z_err_mat = z.z_error[i_per] if z.z_error is not None else None
                row = {
                    "Station":    idx,
                    "StaID2":     idx,
                    "owner":      str(mt.station),
                    "ftype":      None,
                    "X":          float(getattr(mt, "east", 0.0)) / 1000.0,
                    "Y":          float(getattr(mt, "north", 0.0)) / 1000.0,
                    "Freq[Hz]":   1.0 / float(period),
                    "Period[s]":  float(period),
                }
                for ii, mode in enumerate(("Zxx", "Zxy", "Zyx", "Zyy")):
                    i_row, i_col = ii // 2, ii % 2
                    row[f"Re({mode})"] = float(z_mat[i_row, i_col].real)
                    row[f"Im({mode})"] = float(z_mat[i_row, i_col].imag)
                    err = float(z_err_mat[i_row, i_col]) if z_err_mat is not None else np.nan
                    row[f"Re({mode})_SD"] = err
                    row[f"Im({mode})_SD"] = err
                records.append(row)
        df = pd.DataFrame(records)
        return cls(df, units=_UNIT_MODEM, phase_convention="+")

    # exporters 

    def to_file(self,
                filepath: PathLike,
                fill_nan_with: float = NODATA_VAL) -> None:
        """Write the data out as a FEMTIC ``observe.dat`` file.

        The output is in ``"ohm"`` units and the ``exp(-i ω t)``
        convention regardless of the current state of the object; the
        unit and sign conversions are applied on the way out.

        :param filepath: Destination path.
        :type filepath: str or pathlib.Path
        :param fill_nan_with: Value used for NaN entries in observed
            or SD columns, defaults to :data:`NODATA_VAL`.
        :type fill_nan_with: float, optional
        """
        out = self.to_units(_UNIT_OHM)._df.copy()
        # Flip imag signs back to FEMTIC's exp(-iωt) convention.
        for mode in MODES:
            if f"Im({mode})" in out.columns:
                out[f"Im({mode})"] = -out[f"Im({mode})"]
        out = out.fillna(fill_nan_with)
        stations = out["Station"].unique()
        with open(filepath, "w") as f:
            f.write(f"MT {len(stations)}\n")
            for sta_id in stations:
                rows = out[out["Station"] == sta_id]
                x = rows["X"].iloc[0]
                y = rows["Y"].iloc[0]
                sta2 = int(rows["StaID2"].iloc[0])
                f.write(f"{int(sta_id)} {sta2} {x:.6f} {y:.6f}\n")
                f.write(f"{len(rows)}\n")
                for _, r in rows.sort_values("Freq[Hz]").iterrows():
                    f.write(
                        f"{r['Freq[Hz]']:.6f} "
                        f"{r['Re(Zxx)']:.4e} {r['Im(Zxx)']:.4e} "
                        f"{r['Re(Zxy)']:.4e} {r['Im(Zxy)']:.4e} "
                        f"{r['Re(Zyx)']:.4e} {r['Im(Zyx)']:.4e} "
                        f"{r['Re(Zyy)']:.4e} {r['Im(Zyy)']:.4e} "
                        f"{r['Re(Zxx)_SD']:.4e} {r['Im(Zxx)_SD']:.4e} "
                        f"{r['Re(Zxy)_SD']:.4e} {r['Im(Zxy)_SD']:.4e} "
                        f"{r['Re(Zyx)_SD']:.4e} {r['Im(Zyx)_SD']:.4e} "
                        f"{r['Re(Zyy)_SD']:.4e} {r['Im(Zyy)_SD']:.4e}\n"
                    )
            f.write("END\n")

    def to_modem(self, filepath: PathLike, **kwargs) -> None:
        """Write the data as a ModEM ``.data`` impedance file.

        The output is in ``"mV/km/nT"`` units and the ``exp(+i ω t)``
        convention, applied automatically.

        :param filepath: Destination path.
        :type filepath: str or pathlib.Path
        """
        out = self.to_units(_UNIT_MODEM)._df.copy()
        with open(filepath, "w") as f:
            f.write("# Written by FemticData.to_modem\n")
            f.write("# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) "
                    "Component Real Imag Error\n")
            f.write("> Full_Impedance\n> exp(+i\\omega t)\n> [mV/km]/[nT]\n")
            f.write("> 0.00\n> 0.00 0.00\n")
            n_per = out["Period[s]"].nunique()
            n_sta = out["Station"].nunique()
            f.write(f"> {n_per} {n_sta}\n")
            for _, row in out.sort_values(["Period[s]", "Station"]).iterrows():
                # ModEM: X is north, Y is east — swap from FEMTIC's east/north.
                x_m = float(row["Y"]) * 1000.0
                y_m = float(row["X"]) * 1000.0
                code = row.get("owner") or str(int(row["Station"]))
                for mode in MODES:
                    err = row.get(f"Re({mode})_SD", np.nan)
                    re_v = row.get(f"Re({mode})", np.nan)
                    im_v = row.get(f"Im({mode})", np.nan)
                    if (pd.isna(err) or err <= 0 or
                            pd.isna(re_v) or pd.isna(im_v)):
                        continue
                    f.write(
                        f"{row['Period[s]']:.6E} {code} "
                        f"{0.0:.4f} {0.0:.4f} "
                        f"{x_m:.3f} {y_m:.3f} {0.0:.3f} "
                        f"{mode.upper()} {re_v:.6E} {im_v:.6E} {err:.6E}\n"
                    )

    def to_mt_data(self):
        """Convert to an mtpy :class:`mtpy.MTData` object.

        :return: An MTData instance with one MT object per station.
        :rtype: mtpy.MTData

        :raises ImportError: If mtpy is not installed.
        """

        # round-trip through a long DataFrame matching MTData.from_dataframe()'s expected schema.
        out = self.to_units(_UNIT_MODEM)._df.copy()
        records = []
        for _, row in out.iterrows():
            for mode in MODES:
                err = row.get(f"Re({mode})_SD", np.nan)
                if pd.isna(err) or err <= 0:
                    continue
                records.append({
                    "station":  row.get("owner") or str(int(row["Station"])),
                    "period":   row["Period[s]"],
                    "east":     float(row["X"]) * 1000.0,
                    "north":    float(row["Y"]) * 1000.0,
                    "component": mode.lower(),
                    "real":     row[f"Re({mode})"],
                    "imag":     row[f"Im({mode})"],
                    "error":    err,
                })
        long_df = pd.DataFrame(records)
        mt_data = MTData()
        mt_data.from_dataframe(long_df)
        return mt_data

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying wide DataFrame.

        :return: A defensive copy of the internal DataFrame.
        :rtype: pandas.DataFrame
        """
        return self._df.copy()

    # transformations 

    def to_units(self, units: str) -> "FemticData":
        """Return a copy of the data in the requested units.

        :param units: ``"ohm"`` or ``"mV/km/nT"``.
        :type units: str

        :return: A new :class:`FemticData` instance.
        :rtype: FemticData
        """
        _check_units(units)
        if units == self._units:
            return FemticData(self._df.copy(), units=self._units,
                                phase_convention=self._phase)
        df = self._df.copy()
        value_cols = [c for c in df.columns if "(" in c]
        if units == _UNIT_MODEM and self._units == _UNIT_OHM:
            df[value_cols] = df[value_cols] * OHM_TO_MVKMNT
        elif units == _UNIT_OHM and self._units == _UNIT_MODEM:
            df[value_cols] = df[value_cols] / OHM_TO_MVKMNT
        return FemticData(df, units=units, phase_convention=self._phase)

    def mask_nodata(self, nodata_val: float = NODATA_VAL,
                    atol: float = 1e-9) -> "FemticData":
        """Return a copy with FEMTIC sentinels replaced by NaN.

        Operates only on observed-value and SD columns; Cal/Res columns
        are not present in :class:`FemticData` so this affects every
        numeric column that contains a parenthesis in its name.

        :param nodata_val: Sentinel value to replace, defaults to
            :data:`NODATA_VAL`.
        :type nodata_val: float, optional
        :param atol: Absolute tolerance for matching the sentinel,
            defaults to ``1e-9``.
        :type atol: float, optional

        :return: A new :class:`FemticData` instance.
        :rtype: FemticData
        """
        df = self._df.copy()
        for col in df.columns:
            if "(" in col and pd.api.types.is_numeric_dtype(df[col]):
                mask = np.isclose(df[col].to_numpy(dtype=float), nodata_val, atol=atol)
                df.loc[mask, col] = np.nan
        return FemticData(df, units=self._units, phase_convention=self._phase)

    def check_error_floor(self, floor_pct: Optional[float] = None,
                          floor_type: str = "off_diagonal_geometric",
                          tol: float = 1e-3,
                          verbose: bool = False) -> dict:
        """Convenience wrapper for :func:`check_error_floor` on this data.

        :param floor_pct: Expected floor fraction (e.g. ``0.05``). If
            None (default), infer from the data without pass/fail.
        :type floor_pct: float, optional
        :param floor_type: Floor convention. See :func:`check_error_floor`.
        :type floor_type: str, optional
        :param tol: Relative tolerance, defaults to ``1e-3``.
        :type tol: float, optional
        :param verbose: Print a summary if True.
        :type verbose: bool, optional

        :return: Same dict as :func:`check_error_floor`.
        :rtype: dict
        """
        return check_error_floor(self, floor_pct=floor_pct,
                                 floor_type=floor_type, tol=tol,
                                 verbose=verbose)

    def filter_by_relative_error(self,
                                 percent: float,
                                 component: Optional[str] = None,
                                 drop_empty_rows: bool = True) -> "FemticData":
        """Mask observations whose error exceeds ``percent``\\% of ``|Z|``.

        For each affected component, the ``Re``, ``Im``, ``Re_SD`` and
        ``Im_SD`` columns are set to NaN on every row where
        ``SD / |Z| · 100 > percent``. Rows themselves are preserved so
        unaffected components on the same (station, period) sample
        remain usable; rows where every mode ends up missing are
        dropped if ``drop_empty_rows`` is True.

        Points already marked as no-data (``SD ≤ 0`` or NaN, including
        FEMTIC's ``-9.999e-05`` sentinel) fail the threshold by
        definition and are masked.

        :param percent: Threshold percentage. A point is kept only if
            ``SD / |Z| * 100 ≤ percent``.
        :type percent: float
        :param component: If given, only that mode is filtered;
            case-insensitive (``"Zxy"``, ``"ZXY"``, and ``"zxy"`` all
            match). If None (default), every mode is filtered with the
            same threshold.
        :type component: str, optional
        :param drop_empty_rows: If True (default), drop rows where
            every mode is NaN after filtering.
        :type drop_empty_rows: bool, optional

        :return: A new :class:`FemticData` instance.
        :rtype: FemticData

        :raises ValueError: If ``component`` is given and does not
            match any known mode.
        """
        df = self._df.copy()
        if component is not None:
            requested = component.lower()
            match = next((m for m in MODES if m.lower() == requested), None)
            if match is None:
                raise ValueError(
                    f"Unknown component {component!r}. "
                    f"Valid modes (case-insensitive): {MODES}."
                )
            modes_to_filter = (match,)
        else:
            modes_to_filter = MODES

        for mode in modes_to_filter:
            re_col, im_col = f"Re({mode})", f"Im({mode})"
            sd_re, sd_im = f"Re({mode})_SD", f"Im({mode})_SD"
            if re_col not in df.columns or sd_re not in df.columns:
                continue
            amp = np.hypot(df[re_col].to_numpy(dtype=float),
                           df[im_col].to_numpy(dtype=float))
            sd = df[sd_re].to_numpy(dtype=float)
            # Treat SD ≤ 0 / NaN (incl. the -9.999e-05 sentinel) as failing.
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_pct = np.where(sd > 0, (sd / amp) * 100.0, np.inf)
            fail = ~(rel_pct <= percent)  # also True for NaN/inf
            cols = [c for c in (re_col, im_col, sd_re, sd_im)
                    if c in df.columns]
            df.loc[fail, cols] = np.nan

        if drop_empty_rows:
            any_present = pd.Series(False, index=df.index)
            for mode in MODES:
                col = f"Re({mode})"
                if col in df.columns:
                    any_present |= df[col].notna()
            df = df.loc[any_present].reset_index(drop=True)

        return FemticData(df, units=self._units,
                          phase_convention=self._phase)

    def filter_by_relative_error_per_component(
            self,
            percents: Iterable[float] = (200.0, 100.0, 100.0, 200.0),
            components: Iterable[str] = ("Zxx", "Zxy", "Zyx", "Zyy"),
            drop_empty_rows: bool = True) -> "FemticData":
        """Apply per-mode relative-error thresholds in one call.

        Equivalent to calling :meth:`filter_by_relative_error` once for
        each ``(component, percent)`` pair, with the empty-row drop
        deferred until after every component has been processed.

        :param percents: One threshold per ``components`` entry.
            Defaults to ``(200, 100, 100, 200)`` — looser on the
            diagonal (Zxx, Zyy) which is typically noisier, tighter on
            the off-diagonal (Zxy, Zyx).
        :type percents: iterable[float], optional
        :param components: Components to filter, in the same order as
            ``percents``. Defaults to ``("Zxx", "Zxy", "Zyx", "Zyy")``.
        :type components: iterable[str], optional
        :param drop_empty_rows: If True (default), drop rows where
            every mode is NaN after the full per-component pass.
        :type drop_empty_rows: bool, optional

        :return: A new :class:`FemticData` instance.
        :rtype: FemticData

        :raises ValueError: If ``percents`` and ``components`` have
            different lengths.
        """
        percents = list(percents)
        components = list(components)
        if len(percents) != len(components):
            raise ValueError(
                f"percents has {len(percents)} entries but components "
                f"has {len(components)} — they must be the same length."
            )
        out = self
        for comp, pct in zip(components, percents):
            out = out.filter_by_relative_error(percent=pct,
                                               component=comp,
                                               drop_empty_rows=False)
        if drop_empty_rows:
            df = out._df.copy()
            any_present = pd.Series(False, index=df.index)
            for mode in MODES:
                col = f"Re({mode})"
                if col in df.columns:
                    any_present |= df[col].notna()
            df = df.loc[any_present].reset_index(drop=True)
            out = FemticData(df, units=out._units,
                             phase_convention=out._phase)
        return out

    # properties -

    @property
    def units(self) -> str:
        """Current units of the impedance columns (``"ohm"`` or ``"mV/km/nT"``)."""
        return self._units

    @property
    def phase_convention(self) -> str:
        """Phase convention (``"+"`` or ``"-"``)."""
        return self._phase

    @property
    def stations(self) -> np.ndarray:
        """Sorted array of unique station ids."""
        if self._df.empty:
            return np.array([], dtype=int)
        return np.sort(self._df["Station"].unique())

    @property
    def n_stations(self) -> int:
        """Number of unique stations."""
        return int(self._df["Station"].nunique()) if not self._df.empty else 0

    @property
    def periods(self) -> np.ndarray:
        """Sorted array of unique periods in seconds."""
        if self._df.empty:
            return np.array([], dtype=float)
        return np.sort(self._df["Period[s]"].unique())

    @property
    def frequencies(self) -> np.ndarray:
        """Sorted array of unique frequencies in Hz (descending period)."""
        return 1.0 / self.periods[::-1]

    @property
    def n_periods(self) -> int:
        """Number of unique periods."""
        return int(self._df["Period[s]"].nunique()) if not self._df.empty else 0

    @property
    def station_coords(self) -> pd.DataFrame:
        """Per-station coordinate table with columns ``Station``, ``X``, ``Y``."""
        if self._df.empty:
            return pd.DataFrame(columns=["Station", "X", "Y"])
        return (self._df[["Station", "X", "Y"]]
                .drop_duplicates(subset=["Station"])
                .reset_index(drop=True))

    # plotting -

    def plot_station_map(self, ax: Optional[plt.Axes] = None, name_dict: Optional[dict] = None, show: bool = False) -> plt.Figure:
        """Plot the station distribution.

        :param ax: Existing axis, defaults to None.
        :type ax: matplotlib.axes.Axes, optional
        :param name_dict: Optional ``station_id -> display_name``
            mapping for labels, defaults to None.
        :type name_dict: dict, optional
        :param show: Whether to call ``plt.show()`` before returning,
            defaults to False.
        :type show: bool, optional

        :return: The Matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        coords = self.station_coords
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 8))
        else:
            fig = ax.figure
        ax.scatter(coords["Y"], coords["X"], color="tab:blue", s=60, edgecolor="k", zorder=3)
        for _, row in coords.iterrows():
            sid = int(row["Station"])
            label = name_dict.get(sid, sid) if name_dict else sid
            ax.annotate(str(label), (row["Y"], row["X"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("Easting (Y)")
        ax.set_ylabel("Northing (X)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"FEMTIC station map ({self.n_stations} stations)")
        if show:
            plt.show()
        return fig

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (f"FemticData(n_stations={self.n_stations}, "
                f"n_periods={self.n_periods}, units={self._units!r})")


# FemticResponses — wraps result_*_iter*.csv

class FemticResponses:
    """Container for FEMTIC ``result_<freq>_iter<iter>.csv`` outputs.

    The ``_Res`` columns in these files are **already error-normalized**
    residuals — that is, ``Res = (Obs - Cal) / SD``. The RMS of the
    ``_Res`` columns directly gives the standard misfit statistic; do
    not divide by ``_SD`` again.

    :param dataframe: A DataFrame containing the FEMTIC results, with
        one row per (StaID, Freq[Hz]).
    :type dataframe: pandas.DataFrame, optional
    :param units: Unit of the value columns. Defaults to ``"ohm"``.
        Note ``_Res`` columns are dimensionless and never rescaled.
    :type units: str, optional
    :param iteration: The iteration number these results correspond
        to, used for labelling. Defaults to None.
    :type iteration: int, optional
    """

    def __init__(self, dataframe: Optional[pd.DataFrame] = None,
                 units: str = _UNIT_OHM,
                 iteration: Optional[int] = None,
                 phase_convention: str = "+"):
        self._df = dataframe.copy() if dataframe is not None else pd.DataFrame()
        self._units = _check_units(units)
        self._iteration = iteration
        if phase_convention not in ("+", "-"):
            raise ValueError("phase_convention must be '+' or '-'.")
        self._phase = phase_convention
        self.logger = logger
        # Optional mapping ``station_id -> display_name`` used by the
        # plotting methods. Set directly or via
        # :meth:`FemticInversion.attach_names`.
        self.name_dict: dict = {}

    # alternative constructors
    @classmethod
    def from_directory(cls, directory: PathLike,
                       n_freqs: Optional[int] = None,
                       iteration: int = 0,
                       units: str = _UNIT_OHM,
                       phase_convention: str = "+") -> "FemticResponses":
        """Read ``result_<freq>_iter<iteration>.csv`` files into a single object.

        FEMTIC's CSVs store impedances in the ``exp(-iωt)`` convention.
        This constructor flips the sign of every ``Im(...)`` column by
        default so the in-memory object is in the ``exp(+iωt)``
        convention used by mtpy and ModEM — matching what
        :meth:`FemticData.from_file` already does for ``observe.dat``.
        Pass ``phase_convention="-"`` to suppress the flip and keep the
        raw CSV values.

        :param directory: Directory containing the CSVs.
        :type directory: str or pathlib.Path
        :param n_freqs: Number of frequencies (files ``result_0`` ...
            ``result_<n_freqs-1>``). If None (default), every file
            matching ``result_*_iter<iteration>.csv`` is loaded.
        :type n_freqs: int, optional
        :param iteration: Iteration number to read, defaults to 0.
        :type iteration: int, optional
        :param units: Initial units, defaults to ``"ohm"``.
        :type units: str, optional
        :param phase_convention: ``"+"`` (default) or ``"-"`` — see
            above.
        :type phase_convention: str, optional

        :return: A new :class:`FemticResponses` instance.
        :rtype: FemticResponses

        :raises FileNotFoundError: If no matching files are present.
        """
        directory = pathlib.Path(directory)
        if n_freqs is None:
            paths = sorted(directory.glob(f"result_*_iter{iteration}.csv"), key=lambda p: int(re.search(r"result_(\d+)_", p.name).group(1)))
        else:
            paths = [directory / f"result_{i}_iter{iteration}.csv" for i in range(n_freqs)]
            paths = [p for p in paths if p.exists()]

        if not paths:
            raise FileNotFoundError(f"No result_<freq>_iter{iteration}.csv in {directory}")

        frames = []
        for fp in paths:
            _remove_trailing_comma(fp)
            df = pd.read_csv(fp, index_col=None, header=1, usecols=range(0, 34))
            frames.append(df)

        responses = pd.concat(frames, axis=0, ignore_index=True)
        responses.columns = responses.columns.str.strip()
        responses["Period[s]"] = 1.0 / responses["Freq[Hz]"]
        responses["iter"] = iteration
        if "StaID" in responses.columns:
            responses["StaID"] = responses["StaID"].astype(int)

        # FEMTIC writes Z under exp(-iωt). Flip Im columns so the
        # in-memory object matches FemticData (which already flips on
        # read of observe.dat).
        if phase_convention == "+":
            for mode in MODES:
                for suf in ("_Obs", "_Cal", "_Res"):
                    col = f"Im({mode}){suf}"
                    if col in responses.columns:
                        responses[col] = -responses[col]

        obj = cls(responses, units=_UNIT_OHM, iteration=iteration,
                  phase_convention=phase_convention)
        if units == _UNIT_MODEM:
            obj = obj.to_units(_UNIT_MODEM)
        return obj

    # transformations 
    def to_units(self, units: str) -> "FemticResponses":
        """Return a copy with impedance columns in the requested units.

        ``_Res`` columns are dimensionless (error-normalized) and never
        rescaled by this method.

        :param units: ``"ohm"`` or ``"mV/km/nT"``.
        :type units: str

        :return: A new :class:`FemticResponses` instance.
        :rtype: FemticResponses
        """
        _check_units(units)
        if units == self._units:
            return FemticResponses(self._df.copy(), units=self._units,
                                   iteration=self._iteration,
                                   phase_convention=self._phase)
        df = self._df.copy()
        value_cols = [c for c in df.columns if c.startswith(("Re(", "Im(")) and "_Res" not in c]
        if units == _UNIT_MODEM and self._units == _UNIT_OHM:
            df[value_cols] = df[value_cols] * OHM_TO_MVKMNT
        elif units == _UNIT_OHM and self._units == _UNIT_MODEM:
            df[value_cols] = df[value_cols] / OHM_TO_MVKMNT
        return FemticResponses(df, units=units, iteration=self._iteration,
                               phase_convention=self._phase)

    def to_phase_convention(self, convention: str) -> "FemticResponses":
        """Return a copy with imaginary-part signs flipped if needed.

        Going from ``"+"`` (mtpy / ModEM, ``exp(+iωt)``) to ``"-"``
        (FEMTIC native, ``exp(-iωt)``) or vice versa negates every
        ``Im(...)`` column — including ``_Obs``, ``_Cal`` and ``_Res``.
        Real-part columns and standard-deviation columns are left
        untouched (SDs are always positive).

        :param convention: ``"+"`` or ``"-"``.
        :type convention: str

        :return: A new :class:`FemticResponses` instance.
        :rtype: FemticResponses

        :raises ValueError: If ``convention`` is neither ``"+"`` nor
            ``"-"``.
        """
        if convention not in ("+", "-"):
            raise ValueError("convention must be '+' or '-'.")
        if convention == self._phase:
            return FemticResponses(self._df.copy(), units=self._units,
                                   iteration=self._iteration,
                                   phase_convention=convention)
        df = self._df.copy()
        for mode in MODES:
            for suf in ("_Obs", "_Cal", "_Res"):
                col = f"Im({mode}){suf}"
                if col in df.columns:
                    df[col] = -df[col]
        return FemticResponses(df, units=self._units,
                               iteration=self._iteration,
                               phase_convention=convention)

    def with_coords(self, data: FemticData) -> "FemticResponses":
        """Attach ``X`` / ``Y`` station coordinates from a :class:`FemticData`.

        :param data: The :class:`FemticData` to pull coordinates from.
            The join key is ``StaID`` (responses) <-> ``Station`` (data).
        :type data: FemticData

        :return: A new :class:`FemticResponses` with ``Station``,
            ``X``, ``Y`` columns added.
        :rtype: FemticResponses
        """
        coords = data.station_coords
        out = self._df.merge(coords, left_on="StaID", right_on="Station", how="left")
        return FemticResponses(out, units=self._units,
                               iteration=self._iteration,
                               phase_convention=self._phase)

    def add_appres_phase(self) -> "FemticResponses":
        """Return a copy with apparent-resistivity and phase columns added.

        For each ``_Obs`` and ``_Cal`` suffix, the function adds
        ``AppRes_<mode><suffix>`` (Ω·m) and ``Phase_<mode><suffix>``
        (deg) using the standard MT formula
        ``ρ_a = |Z|² / (ω μ₀)``, which requires Z in SI ohms. FEMTIC's
        ``"ohm"`` units are already SI; ``"mV/km/nT"`` values are
        converted back to SI internally by dividing by
        :data:`OHM_TO_MVKMNT`. No state change occurs.

        Error columns ``AppRes_<mode>_Obs_Err`` and
        ``Phase_<mode>_Obs_Err`` are also added, propagated from the
        impedance SD with the first-order expressions

        .. math::

           dZ = \\mathrm{Re}(Z_\\mathrm{err}) \\,/\\, |Z|, \\quad
           \\Delta\\rho_a = 2 \\rho_a\\, dZ, \\quad
           \\Delta\\varphi = dZ \\cdot 180/\\pi \\;\\;[\\mathrm{deg}].

        :return: A new :class:`FemticResponses` with additional columns.
        :rtype: FemticResponses
        """
        # ρ_a = |Z_SI|² / (ω μ₀).  Convert to SI ohms first if needed.
        scale = 1.0 if self._units == _UNIT_OHM else 1.0 / OHM_TO_MVKMNT
        df = self._df.copy()
        omega = 2.0 * np.pi * df["Freq[Hz]"].to_numpy()
        mu = 4.0e-7 * np.pi
        for suf in ("_Obs", "_Cal"):
            for mode in MODES:
                re_col = f"Re({mode}){suf}"
                im_col = f"Im({mode}){suf}"
                if re_col not in df.columns or im_col not in df.columns:
                    continue
                z = scale * (df[re_col].to_numpy() + 1j * df[im_col].to_numpy())
                df[f"AppRes_{mode}{suf}"] = (1.0 / (omega * mu)) * np.abs(z) ** 2
                df[f"Phase_{mode}{suf}"]  = np.rad2deg(np.angle(z))
        # Propagate impedance SD to apparent-resistivity and phase errors
        # using dZ = Re(Z_err) / |Z| with the first-order formulae
        # Δρ_a = 2 ρ_a dZ  and  Δφ = dZ * 180/π  (in degrees).
        for mode in MODES:
            re_col, im_col = f"Re({mode})_Obs", f"Im({mode})_Obs"
            err_col = f"Re({mode})_SD"
            if (re_col not in df.columns or im_col not in df.columns
                    or err_col not in df.columns):
                continue
            z_obs = scale * (df[re_col].to_numpy() + 1j * df[im_col].to_numpy())
            sd    = scale * df[err_col].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                dZ_rel = sd / np.abs(z_obs)
            df[f"AppRes_{mode}_Obs_Err"] = 2.0 * df[f"AppRes_{mode}_Obs"] * dZ_rel
            df[f"Phase_{mode}_Obs_Err"]  = np.rad2deg(dZ_rel)
        return FemticResponses(df, units=self._units,
                               iteration=self._iteration,
                               phase_convention=self._phase)

    def residuals(self, swap_xy: bool = True) -> pd.DataFrame:
        """Return long-form residuals suitable for :class:`RMSAnalysis`.

        Each (station, period) sample contributes up to eight rows: the
        Real and Imag part of each of the four impedance modes, treated
        **independently**. A component is included only when its own
        standard-deviation field (``Re(<mode>)_SD`` for the real part,
        ``Im(<mode>)_SD`` for the imaginary part) is positive — FEMTIC
        flags an unused component with a negative ``_SD`` — and each
        residual is normalized by that same per-component ``_SD``. Real
        and imaginary parts of the same mode are kept or dropped
        independently, so a sample where only one part is flagged still
        contributes its valid part.

        :param swap_xy: If True, output ``X(m)`` and ``Y(m)`` in the
            ModEM convention (X = north, Y = east) by swapping from the
            FEMTIC convention, defaults to True.
        :type swap_xy: bool, optional

        :return: Long-form residuals DataFrame with columns
            ``Period(s)``, ``Code``, ``X(m)``, ``Y(m)``, ``Component``,
            ``Complex_flag``, ``Res``, ``Error``.
        :rtype: pandas.DataFrame
        """
        if "X" not in self._df.columns or "Y" not in self._df.columns:
            raise RuntimeError(
                "Station coordinates missing. Call .with_coords(data) first."
            )
        recs = []
        for _, row in self._df.iterrows():
            if swap_xy:
                x_out, y_out = row["Y"] * 1000.0, row["X"] * 1000.0
            else:
                x_out, y_out = row["X"] * 1000.0, row["Y"] * 1000.0
            code = _format_code(row.get("Station", row.get("StaID")))
            for mode in MODES:
                # Real and imaginary parts are independent data: each is kept
                # only if its OWN _SD is positive (a negative _SD is FEMTIC's
                # no-data flag) and is normalized by that same per-component _SD.
                re_sd = row.get(f"Re({mode})_SD", np.nan)
                im_sd = row.get(f"Im({mode})_SD", np.nan)
                re_o = row.get(f"Re({mode})_Obs", np.nan)
                im_o = row.get(f"Im({mode})_Obs", np.nan)
                re_c = row.get(f"Re({mode})_Cal", np.nan)
                im_c = row.get(f"Im({mode})_Cal", np.nan)
                if (pd.notna(re_sd) and re_sd > 0
                        and pd.notna(re_o) and pd.notna(re_c)):
                    recs.append({
                        "Period(s)":    row["Period[s]"],
                        "Code":         code,
                        "X(m)":         x_out,
                        "Y(m)":         y_out,
                        "Component":    mode.upper(),
                        "Complex_flag": "Real",
                        "Res":          re_o - re_c,
                        "Error":        re_sd,
                    })
                if (pd.notna(im_sd) and im_sd > 0
                        and pd.notna(im_o) and pd.notna(im_c)):
                    recs.append({
                        "Period(s)":    row["Period[s]"],
                        "Code":         code,
                        "X(m)":         x_out,
                        "Y(m)":         y_out,
                        "Component":    mode.upper(),
                        "Complex_flag": "Imag",
                        "Res":          im_o - im_c,
                        "Error":        im_sd,
                    })
        return pd.DataFrame(recs).sort_values(
            ["Code", "Period(s)"]).reset_index(drop=True)

    def rms_analysis(self, normalize_by_error: bool = True, swap_xy: bool = True,
                     femtic_convention: bool = False) -> "RMSAnalysis":
        """Compute an :class:`RMSAnalysis` over this object's residuals.

        :param normalize_by_error: If True (default), divide each
            residual by its error before computing RMS.
        :type normalize_by_error: bool, optional
        :param swap_xy: Forwarded to :meth:`residuals`.
        :type swap_xy: bool, optional
        :param femtic_convention: If True, make :attr:`RMSAnalysis.overall`
            reproduce FEMTIC's reported ``femtic.cnv`` RMS, i.e.
            ``sqrt(2 * sum(norm**2) / N)`` where ``N`` counts every component
            slot (all four modes' real and imaginary parts on every
            station/frequency row, including the no-data ones). The factor of
            two is FEMTIC's complex-data normalization. The per-station /
            per-period breakdowns are unaffected and remain the plain
            normalized RMS. Defaults to False.
        :type femtic_convention: bool, optional

        :return: A new :class:`RMSAnalysis` instance.
        :rtype: RMSAnalysis
        """
        n_modes = sum(1 for m in MODES if f"Re({m})_Obs" in self._df.columns)
        n_total = int(len(self._df) * n_modes * 2)
        return RMSAnalysis(self.residuals(swap_xy=swap_xy),
                           normalize_by_error=normalize_by_error,
                           name_dict=getattr(self, "name_dict", None),
                           femtic_convention=femtic_convention,
                           n_total=n_total)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame.

        :rtype: pandas.DataFrame
        """
        return self._df.copy()

    @property
    def units(self) -> str:
        """Current units of the value columns."""
        return self._units

    @property
    def phase_convention(self) -> str:
        """Time-harmonic convention of stored imag components (``"+"`` or ``"-"``)."""
        return self._phase

    @property
    def iteration(self) -> Optional[int]:
        """Iteration number these results came from."""
        return self._iteration

    @property
    def stations(self) -> np.ndarray:
        """Sorted array of unique station ids."""
        col = _resolve_station_col(self._df)
        return np.sort(self._df[col].unique())

    @property
    def n_stations(self) -> int:
        """Number of unique stations."""
        col = _resolve_station_col(self._df)
        return int(self._df[col].nunique())

    @property
    def periods(self) -> np.ndarray:
        """Sorted array of unique periods in seconds."""
        return np.sort(self._df["Period[s]"].unique())

    @property
    def n_periods(self) -> int:
        """Number of unique periods."""
        return int(self._df["Period[s]"].nunique())

    # plotting -

    def plot_response(self, station: Optional[Union[int, Iterable[int]]] = None,
                      kind: str = "z",
                      diag_ylim=(1e-1, 1e5), off_ylim=(1e-1, 1e5),
                      xlim: Optional[tuple] = None,
                      include_rms_in_title: bool = True,
                      phase_overlap: bool = True,
                      show: bool = False) -> list:
        """Plot impedance or apparent resistivity / phase responses.

        :param station: A single station id, an iterable of ids, or
            None to plot every station, defaults to None.
        :type station: int, iterable, or None, optional
        :param kind: ``"z"`` for Real/Imag impedance, ``"appres"`` for
            apparent resistivity / phase. Defaults to ``"z"``.
        :type kind: str, optional
        :param diag_ylim: Y-limits for the on-diagonal apparent
            resistivity plot, defaults to ``(1e-1, 1e5)``.
        :type diag_ylim: tuple[float, float], optional
        :param off_ylim: Y-limits for the off-diagonal apparent
            resistivity plot, defaults to ``(1e-1, 1e5)``.
        :type off_ylim: tuple[float, float], optional
        :param xlim: Period-axis limits ``(lo, hi)`` in seconds applied
            to every subplot. If None (default), the full period range
            of this :class:`FemticResponses` is used with ~10%
            log-space padding so every station's figure shares the
            same x-axis.
        :type xlim: tuple[float, float], optional
        :param include_rms_in_title: If True (default), compute the
            per-station RMS via :meth:`rms_analysis` and append it to
            each figure's suptitle.
        :type include_rms_in_title: bool, optional
        :param phase_overlap: For ``kind="appres"`` only — if True
            (default), shift the Zyx phase by +180° so it overlaps the
            Zxy phase in the same quadrant (the standard MT plotting
            convention). The legend label becomes ``"Zyx + 180 cal"``
            and ``"Zyx + 180 obs"`` when the shift is active.
        :type phase_overlap: bool, optional
        :param show: Whether to call ``plt.show()`` before returning,
            defaults to False.
        :type show: bool, optional

        :return: List of created figures.
        :rtype: list[matplotlib.figure.Figure]

        :raises ValueError: If ``kind`` is not ``"z"`` or ``"appres"``.
        """
        if kind not in ("z", "appres"):
            raise ValueError("kind must be 'z' or 'appres'.")
        if station is None:
            station_ids = self.stations
        elif np.isscalar(station):
            station_ids = [station]
        else:
            station_ids = list(station)

        if xlim is None:
            periods = self.periods
            if periods.size > 0:
                log_pad = 0.04  # ~10% in log space
                xlim = (float(periods.min()) * 10.0 ** (-log_pad),
                        float(periods.max()) * 10.0 ** log_pad)

        rms_per_station = {}
        if include_rms_in_title:
            try:
                ps = self.rms_analysis().per_station
                rms_per_station = {row["Code"]: float(row["RMS"])
                                   for _, row in ps.iterrows()}
            except Exception:
                # No coordinates attached, or no valid residuals.
                # Silently fall back to RMS-free titles.
                pass

        sta_col = _resolve_station_col(self._df)
        if kind == "z":
            return self._plot_z(station_ids, sta_col, rms_per_station,
                                xlim, show)
        else:
            return self._plot_appres(station_ids, sta_col,
                                     diag_ylim, off_ylim,
                                     rms_per_station, phase_overlap,
                                     xlim, show)

    def _plot_z(self, station_ids, sta_col, rms_per_station, xlim, show):
        figs = []
        # Use a display copy in mV/km/nT for readability.
        disp = self.to_units(_UNIT_MODEM)._df
        for sta in station_ids:
            sub = disp[disp[sta_col] == sta].sort_values("Period[s]")
            if sub.empty:
                continue
            fig, ((ax_re_off, ax_re_diag), (ax_im_off, ax_im_diag)) = plt.subplots(
                2, 2, figsize=(13, 7), sharex=True)
            rms_val = rms_per_station.get(_format_code(sta))
            fig.suptitle(_station_title(sta, self._iteration,
                                        self.name_dict, rms=rms_val))
            per = sub["Period[s]"].to_numpy()
            for mode in MODES:
                color = COMPONENT_COLORS[mode]
                ax_re = ax_re_off if mode in ("Zxy", "Zyx") else ax_re_diag
                ax_im = ax_im_off if mode in ("Zxy", "Zyx") else ax_im_diag
                re_cal = sub[f"Re({mode})_Cal"].to_numpy()
                im_cal = sub[f"Im({mode})_Cal"].to_numpy()
                re_obs = sub[f"Re({mode})_Obs"].to_numpy()
                im_obs = sub[f"Im({mode})_Obs"].to_numpy()
                sd = sub[f"Re({mode})_SD"].to_numpy()
                mask = sd > 0
                ax_re.scatter(  per, re_cal, facecolors="none",
                                edgecolors=color, marker="o", label=f"{mode} cal")
                ax_im.scatter(  per, im_cal, facecolors="none",
                                edgecolors=color, marker="o", label=f"{mode} cal")
                if mask.any():
                    ax_re.errorbar( per[mask], re_obs[mask], yerr=sd[mask],
                                    fmt="x", color=color, linewidth=0,
                                    capsize=3, elinewidth=1, label=f"{mode} obs")
                    ax_im.errorbar( per[mask], im_obs[mask], yerr=sd[mask],
                                    fmt="x", color=color, linewidth=0,
                                    capsize=3, elinewidth=1, label=f"{mode} obs")
            for ax, ylabel in [ (ax_re_off,  "Re(Z) off-diagonal"),
                                (ax_im_off,  "Im(Z) off-diagonal"),
                                (ax_re_diag, "Re(Z) on-diagonal"),
                                (ax_im_diag, "Im(Z) on-diagonal")]:
                ax.set_xscale("log")
                ax.set_ylabel(ylabel + "  [mV/km/nT]")
                ax.grid(True, which="both", alpha=0.3)
            if xlim is not None:
                # sharex=True propagates this to all four axes
                ax_re_off.set_xlim(*xlim)
            ax_im_off.set_xlabel("Period [s]")
            ax_im_diag.set_xlabel("Period [s]")
            ax_re_off.legend(fontsize=8, ncol=2)
            ax_re_diag.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            figs.append(fig)
        if show:
            plt.show()
        return figs

    def _plot_appres(self, station_ids, sta_col, diag_ylim, off_ylim,
                     rms_per_station, phase_overlap, xlim, show):
        ar = self.add_appres_phase()._df
        figs = []
        for sta in station_ids:
            sub = ar[ar[sta_col] == sta].sort_values("Period[s]")
            if sub.empty:
                continue
            fig, ((ax_ar_off, ax_ar_diag), (ax_ph_off, ax_ph_diag)) = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
            rms_val = rms_per_station.get(_format_code(sta))
            fig.suptitle(_station_title(sta, self._iteration,
                                        self.name_dict, rms=rms_val))
            per = sub["Period[s]"].to_numpy()
            for mode in MODES:
                color = COMPONENT_COLORS[mode]
                if mode in ("Zxy", "Zyx"):
                    ax_ar, ax_ph, ylim = ax_ar_off, ax_ph_off, off_ylim
                else:
                    ax_ar, ax_ph, ylim = ax_ar_diag, ax_ph_diag, diag_ylim

                ph_shift = 180.0 if (phase_overlap and mode == "Zyx") else 0.0
                ph_label = f"{mode} + 180" if ph_shift else mode

                ph_cal = sub[f"Phase_{mode}_Cal"].to_numpy() + ph_shift

                ax_ar.plot(per, sub[f"AppRes_{mode}_Cal"],
                           color=color, label=f"{mode} cal")
                ax_ph.plot(per, ph_cal,
                           color=color, label=f"{ph_label} cal")

                mask = sub[f"Re({mode})_SD"].to_numpy() > 0
                if mask.any():
                    ar_obs = sub[f"AppRes_{mode}_Obs"].to_numpy()[mask]
                    ar_err = sub[f"AppRes_{mode}_Obs_Err"].to_numpy()[mask]
                    ph_obs = sub[f"Phase_{mode}_Obs"].to_numpy()[mask] + ph_shift
                    ph_err = sub[f"Phase_{mode}_Obs_Err"].to_numpy()[mask]
                    ax_ar.errorbar(per[mask], ar_obs, yerr=ar_err,
                                   fmt="x", color=color, linewidth=0,
                                   capsize=3, elinewidth=1, label=f"{mode} obs")
                    ax_ph.errorbar(per[mask], ph_obs, yerr=ph_err,
                                   fmt="x", color=color, linewidth=0,
                                   capsize=3, elinewidth=1,
                                   label=f"{ph_label} obs")
                ax_ar.set_ylim(*ylim)
            for ax in (ax_ar_off, ax_ar_diag):
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_ylabel("App. Resistivity [Ωm]")
                ax.grid(True, which="both", alpha=0.3)
            for ax in (ax_ph_off, ax_ph_diag):
                ax.set_xscale("log")
                ax.set_ylabel("Phase [deg]")
                ax.set_xlabel("Period [s]")
                ax.grid(True, which="both", alpha=0.3)
            ax_ar_off.legend(fontsize=8, ncol=2)
            ax_ar_diag.legend(fontsize=8, ncol=2)
            ax_ph_off.legend(fontsize=8, ncol=2)
            ax_ph_diag.legend(fontsize=8, ncol=2)
            if xlim is not None:
                # sharex=True propagates this to all four axes
                ax_ar_off.set_xlim(*xlim)
            fig.tight_layout()
            figs.append(fig)
        if show:
            plt.show()
        return figs


    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (f"FemticResponses(iter={self._iteration}, "
                f"n_stations={self.n_stations}, "
                f"n_periods={self.n_periods}, units={self._units!r}, "
                f"phase={self._phase!r})")


class FemticConvergence:
    """Container for a ``femtic.cnv`` convergence file.

    FEMTIC may record several ``Retrial#`` rows for the same ``Iter#``
    (an iteration that was repeated). Only the highest ``Retrial#`` of
    each iteration is the accepted result kept on disk, so on
    construction the rows are collapsed to one row per iteration — the
    highest-retrial one. Every convergence statistic and the plot
    therefore reflect the accepted iterations only.

    :param dataframe: A DataFrame with the FEMTIC convergence schema.
    :type dataframe: pandas.DataFrame, optional
    """

    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        df = dataframe.copy() if dataframe is not None else pd.DataFrame()
        self._df = self._keep_last_retrial(df)
        self.logger = logger

    @staticmethod
    def _keep_last_retrial(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the highest ``Retrial#`` row for each ``Iter#``.

        The kept rows are sorted by ``Iter#`` with a fresh index. If the
        ``Iter#`` / ``Retrial#`` columns are absent (or the frame is
        empty), the input is returned unchanged.

        :param df: Raw convergence rows, possibly with multiple retrials
            per iteration.
        :type df: pandas.DataFrame

        :return: One row per iteration (the accepted, highest-retrial one).
        :rtype: pandas.DataFrame
        """
        if df.empty or "Iter#" not in df.columns or "Retrial#" not in df.columns:
            return df
        keep_idx = df.groupby("Iter#")["Retrial#"].idxmax()
        return (df.loc[keep_idx]
                  .sort_values("Iter#")
                  .reset_index(drop=True))

    @classmethod
    def from_file(cls, filepath: Optional[PathLike] = None) -> "FemticConvergence":
        """Read a ``femtic.cnv`` file.

        :param filepath: Path to the file. Defaults to
            ``./femtic.cnv``.
        :type filepath: str or pathlib.Path, optional

        :return: A new :class:`FemticConvergence` instance.
        :rtype: FemticConvergence
        """
        if filepath is None:
            filepath = pathlib.Path.cwd() / "femtic.cnv"
        return cls(pd.read_csv(filepath, sep=r"\s+"))

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame.

        :rtype: pandas.DataFrame
        """
        return self._df.copy()

    @property
    def final_rms(self) -> float:
        """RMS at the last (highest-iteration, highest-retrial) row."""
        return float(self._df.iloc[-1]["RMS"])

    @property
    def best_rms(self) -> float:
        """Minimum RMS achieved across all iterations and retrials."""
        return float(self._df["RMS"].min())

    @property
    def best_iteration(self) -> int:
        """Iteration number at which :attr:`best_rms` was achieved."""
        return int(self._df.loc[self._df["RMS"].idxmin(), "Iter#"])

    @property
    def iterations(self) -> np.ndarray:
        """Sorted unique iteration numbers present."""
        return np.sort(self._df["Iter#"].unique())

    def plot(self, ax: Optional[plt.Axes] = None, title: str = "FEMTIC convergence", show: bool = False) -> plt.Figure:
        """Plot RMS and ancillary convergence quantities.

        :param ax: Existing axis, defaults to None.
        :type ax: matplotlib.axes.Axes, optional
        :param title: Title for the figure, defaults to ``"FEMTIC convergence"``.
        :type title: str, optional
        :param show: Whether to call ``plt.show()`` before returning,
            defaults to False.
        :type show: bool, optional

        :return: The Matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        else:
            fig = ax.figure
        fig.subplots_adjust(right=0.7)
        p0, = ax.plot(self._df["Iter#"], self._df["RMS"], marker="o", label="RMS")
        ax.set(xlabel="Iteration #", ylabel="RMS Misfit", title=title)
        ax.yaxis.label.set_color(p0.get_color())
        ax.tick_params(axis="y", colors=p0.get_color())
        extra_cols = [c for c in ["Alpha", "Damp", "Roughness", "Misfit", "ObjFunc"] if c in self._df.columns]
        for i, col in enumerate(extra_cols):
            tw = ax.twinx()
            tw.spines.right.set_position(("axes", 1.0 + 0.10 * i))
            p, = tw.plot(self._df["Iter#"], self._df[col], "-o", markersize=3, color=f"C{i+1}", alpha=0.5)
            tw.set_ylabel(col)
            tw.yaxis.label.set_color(p.get_color())
            tw.tick_params(axis="y", colors=p.get_color())
        if show:
            plt.show()
        return fig

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        if self._df.empty:
            return "FemticConvergence(empty)"
        return (f"FemticConvergence(n_iterations={len(self.iterations)}, "
                f"best_rms={self.best_rms:.4f} @ iter {self.best_iteration})")


class RMSAnalysis:
    """Aggregated RMS misfit at multiple granularities.

    Computes (on construction) the overall RMS as well as per-station,
    per-period, per-period-and-component, and per-period-component-station
    breakdowns. The breakdowns are exposed as DataFrame properties and
    plotted via the ``plot_*`` methods.

    :param residuals: Long-form residuals DataFrame, typically from
        :meth:`FemticResponses.residuals`. Required columns:
        ``Period(s)``, ``Code``, ``X(m)``, ``Y(m)``, ``Component``,
        ``Res``, ``Error``.
    :type residuals: pandas.DataFrame
    :param normalize_by_error: If True (default), each residual is
        divided by its error before computing RMS. Set False to use the
        raw residuals (rarely useful — included for completeness).
    :type normalize_by_error: bool, optional
    :param femtic_convention: If True, :attr:`overall` reproduces FEMTIC's
        reported ``femtic.cnv`` RMS, ``sqrt(2 * sum(norm**2) / n_total)``:
        a factor of two for FEMTIC's complex-data normalization, and a
        denominator that counts every component slot (``n_total``) rather
        than only the valid residuals. The breakdowns (:attr:`per_station`,
        :attr:`per_period`, ...) are left as the plain normalized RMS.
        Defaults to False.
    :type femtic_convention: bool, optional
    :param n_total: Total number of component slots used as the FEMTIC
        denominator (all four modes' real and imaginary parts on every
        station/frequency row, including no-data slots). Only used when
        ``femtic_convention`` is True; defaults to the number of valid
        residuals when not supplied.
    :type n_total: int, optional
    """

    def __init__(self, residuals: pd.DataFrame, normalize_by_error: bool = True,
                 name_dict: Optional[dict] = None,
                 femtic_convention: bool = False,
                 n_total: Optional[int] = None):
        self._residuals = residuals.copy()
        self.normalize_by_error = normalize_by_error
        # Optional mapping ``station_id -> display_name`` used by plots.
        self.name_dict: dict = dict(name_dict) if name_dict else {}
        self.femtic_convention = femtic_convention
        self.n_total = n_total
        self.logger = logger
        self._compute()

    def _compute(self) -> None:
        df = self._residuals
        if self.normalize_by_error:
            df = df.assign(_norm=df["Res"] / df["Error"])
        else:
            df = df.assign(_norm=df["Res"])
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["_norm"])

        if self.femtic_convention:
            # FEMTIC reports sqrt(2 * sum(norm^2) / N) with N counting every
            # component slot (valid + no-data). Reproduce that exactly.
            n = self.n_total if self.n_total else len(df)
            ssq = float(np.sum(np.asarray(df["_norm"], dtype=float) ** 2))
            self.overall = float(np.sqrt(2.0 * ssq / n)) if n else float("nan")
        else:
            self.overall = _rms(df["_norm"])

        coords = (df[["Code", "X(m)", "Y(m)"]].drop_duplicates(subset=["Code"]).reset_index(drop=True))

        # self.per_station = (df.groupby("Code", as_index=False)["_norm"].agg(RMS=lambda s: _rms(s), n="size").merge(coords, on="Code", how="left"))
        # self.per_station = (df.groupby("Code", as_index=False).agg({"RMS":("_norm", lambda s: _rms(s)), "n":("_norm", "size"), "X(m)":("_norm", coords["X(m)"]), "Y(m)":("_norm", coords["Y(m)"])}))
        self.per_station = (df.groupby(["Code", "X(m)", "Y(m)"], as_index=False).agg(RMS=("_norm", lambda s: _rms(s)), n=("_norm", "size")))

        # self.per_period = (df.groupby("Period(s)", as_index=False)["_norm"].agg(RMS=lambda s: _rms(s), n="size"))
        self.per_period = (df.groupby("Period(s)", as_index=False).agg(RMS=("_norm", lambda s: _rms(s)), n=("_norm", "size")))

        # self.per_period_component = (df.groupby(["Period(s)", "Component"], as_index=False)["_norm"].agg(RMS=lambda s: _rms(s), n="size"))
        self.per_period_component = (df.groupby(["Period(s)", "Component"], as_index=False).agg(RMS=("_norm", lambda s: _rms(s)), n=("_norm", "size")))

        # self.per_period_component_station = (df.groupby(["Period(s)", "Component", "Code"], as_index=False)["_norm"].agg(RMS=lambda s: _rms(s), n="size"))
        self.per_period_component_station = (df.groupby(["Period(s)", "Component", "Code"], as_index=False).agg(RMS=("_norm", lambda s: _rms(s)), n=("_norm", "size")))

        self._df = df


    def summary(self, top_n: int = 5) -> str:
        """Return a multi-line summary string.

        :param top_n: How many worst-fit stations and periods to list,
            defaults to 5.
        :type top_n: int, optional

        :return: A formatted summary suitable for printing.
        :rtype: str
        """
        lines = [f"Overall RMS: {self.overall:.4f}",
                "",
                f"Worst {top_n} stations:",
                (self.per_station.sort_values("RMS", ascending=False)
                                    .head(top_n).to_string(index=False)),
                "",
                "RMS per period:",
                self.per_period.to_string(index=False)]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"RMSAnalysis(overall={self.overall:.4f}, "
                f"n_obs={len(self._df)}, "
                f"normalized={self.normalize_by_error})")

    def plot_station_map(self,
                         ax: Optional[plt.Axes] = None,
                         name_dict: Optional[dict] = None,
                         contour: bool = False,
                         contour_levels=10,
                         contour_filled: bool = False,
                         contour_alpha: float = 0.5,
                         contour_kwargs: Optional[dict] = None,
                         show: bool = False) -> plt.Figure:
        """Plot a station map color-coded by per-station RMS.

        Optionally overlay interpolated RMS contours computed by
        Delaunay triangulation of the station positions.

        :param ax: Existing axis, defaults to None.
        :type ax: matplotlib.axes.Axes, optional
        :param name_dict: Optional ``station_id -> display_name`` mapping
            for labels. Falls back to :attr:`name_dict` set on this
            object if not provided, defaults to None.
        :type name_dict: dict, optional
        :param contour: If True, draw triangulated RMS contours behind
            the station markers, defaults to False.
        :type contour: bool, optional
        :param contour_levels: Number of contour levels or an explicit
            sequence of level values, defaults to ``10``.
        :type contour_levels: int or sequence, optional
        :param contour_filled: If True, draw filled contours
            (``tricontourf``) in addition to line contours, defaults to
            False.
        :type contour_filled: bool, optional
        :param contour_alpha: Alpha for the filled contours, defaults
            to ``0.5``.
        :type contour_alpha: float, optional
        :param contour_kwargs: Extra keyword arguments passed through
            to :func:`matplotlib.pyplot.tricontour`, defaults to None.
        :type contour_kwargs: dict, optional
        :param show: Whether to call ``plt.show()`` before returning,
            defaults to False.
        :type show: bool, optional

        :return: The Matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        if name_dict is None:
            name_dict = getattr(self, "name_dict", None)

        coords = self.per_station.copy()
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 8))
        else:
            fig = ax.figure

        # Contour overlay (behind markers)
        if contour:
            x = coords["Y(m)"].to_numpy(dtype=float)
            y = coords["X(m)"].to_numpy(dtype=float)
            z = coords["RMS"].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if valid.sum() >= 3:
                cargs = dict(levels=contour_levels, cmap="viridis")
                if contour_kwargs:
                    cargs.update(contour_kwargs)
                if contour_filled:
                    ax.tricontourf(x[valid], y[valid], z[valid],
                                   alpha=contour_alpha, **cargs)
                cs = ax.tricontour(x[valid], y[valid], z[valid],
                                   linewidths=1.0, **cargs)
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
            else:
                # Not enough points to triangulate — silently skip.
                pass

        sc = ax.scatter(coords["Y(m)"], coords["X(m)"], c=coords["RMS"],
                        cmap="viridis", s=80, edgecolor="k", zorder=3)
        plt.colorbar(sc, ax=ax, label="RMS")
        for _, row in coords.iterrows():
            code = row["Code"]
            label = (name_dict.get(int(code), code)
                     if (name_dict and code.isdigit()) else code)
            ax.annotate(str(label), (row["Y(m)"], row["X(m)"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
            ax.annotate(f"{row['RMS']:.2f}", (row["Y(m)"], row["X(m)"]),
                        textcoords="offset points", xytext=(6, -8),
                        fontsize=7)
        ax.set_xlabel("Easting (Y(m))")
        ax.set_ylabel("Northing (X(m))")
        ax.set_aspect("equal", adjustable="datalim")
        title = "Station map (color: RMS"
        if contour:
            title += "; contours interpolated"
        title += ")"
        ax.set_title(title)
        if show:
            plt.show()
        return fig

    def plot_per_period(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Figure:
        """Plot total RMS as a function of period (log x).

        :param ax: Existing axis, defaults to None.
        :type ax: matplotlib.axes.Axes, optional
        :param show: Whether to call ``plt.show()`` before returning,
            defaults to False.
        :type show: bool, optional

        :return: The Matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        d = self.per_period.sort_values("Period(s)")
        ax.semilogx(d["Period(s)"], d["RMS"], "-o")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("RMS")
        ax.set_title("RMS per period")
        ax.grid(True, which="both", alpha=0.3)
        if show:
            plt.show()
        return fig

    def plot_per_period_component(self, ax: Optional[plt.Axes] = None, show: bool = False) -> plt.Figure:
        """Plot RMS vs period, separated by impedance component.

        :param ax: Existing axis, defaults to None.
        :type ax: matplotlib.axes.Axes, optional
        :param show: Whether to call ``plt.show()`` before returning,
            defaults to False.
        :type show: bool, optional

        :return: The Matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            fig = ax.figure
        for comp in sorted(self.per_period_component["Component"].unique()):
            sub = (self.per_period_component[self.per_period_component["Component"] == comp].sort_values("Period(s)"))
            ax.semilogx(sub["Period(s)"], sub["RMS"], "-o", label=comp, color=COMPONENT_COLORS.get(comp.capitalize()))
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("RMS")
        ax.set_title("RMS per period, by component")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        if show:
            plt.show()
        return fig

    def plot_per_period_component_station(self,
                                          name_dict: Optional[dict] = None,
                                          legend: bool = False,
                                          show: bool = False) -> plt.Figure:
        """Plot a 2x2 grid of RMS-per-period-per-station, one panel per mode.

        :param name_dict: Optional ``station_id -> display_name`` mapping
            used in the legend. Falls back to :attr:`name_dict` set on
            this object if not provided, defaults to None.
        :type name_dict: dict, optional
        :param legend: If True, draw a station legend on each subplot
            (off by default since networks with many stations make this
            unreadable).
        :type legend: bool, optional
        :param show: Whether to call ``plt.show()`` before returning,
            defaults to False.
        :type show: bool, optional

        :return: The Matplotlib figure.
        :rtype: matplotlib.figure.Figure
        """
        if name_dict is None:
            name_dict = getattr(self, "name_dict", None)

        def display(code):
            if not name_dict:
                return code
            if code.isdigit() and int(code) in name_dict:
                return name_dict[int(code)]
            if code in name_dict:
                return name_dict[code]
            return code

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        codes = sorted(self.per_period_component_station["Code"].unique(),
                       key=lambda s: (len(s), s))
        cmap = plt.get_cmap("tab20")
        code_colors = {c: cmap(i % cmap.N) for i, c in enumerate(codes)}
        for ax, comp in zip(axes.flat, ("ZXX", "ZXY", "ZYX", "ZYY")):
            sub = self.per_period_component_station[
                self.per_period_component_station["Component"] == comp]
            for code, grp in sub.groupby("Code"):
                grp = grp.sort_values("Period(s)")
                ax.semilogx(grp["Period(s)"], grp["RMS"], "-o",
                            color=code_colors[code], markersize=3,
                            linewidth=0.8, alpha=0.7, label=display(code))
            ax.set_title(comp)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_ylabel("RMS")
            if legend:
                ax.legend(fontsize=6, ncol=2, loc="best")
        for ax in axes[1, :]:
            ax.set_xlabel("Period [s]")
        fig.suptitle("RMS per period per station, by component")
        fig.tight_layout()
        if show:
            plt.show()
        return fig


class FemticInversion:
    """A complete FEMTIC inversion run.

    Ties together the input observation, every available iteration's
    response files, and the convergence diagnostics for a single
    inversion. The most common usage is to instantiate via
    :meth:`from_directory`::

        inv = FemticInversion.from_directory("path/to/inversion")
        inv.data                              # FemticData (observe.dat)
        inv.convergence                       # FemticConvergence (femtic.cnv)
        inv.iterations                        # list of available iters
        inv.results[11]                       # FemticResponses for iter 11
        inv.rms_analysis(iteration=11)        # RMSAnalysis for iter 11

    :param root_dir: The inversion directory.
    :type root_dir: str or pathlib.Path
    :param data: The observation, defaults to None.
    :type data: FemticData, optional
    :param results: Mapping ``iteration -> FemticResponses``, defaults
        to an empty dict.
    :type results: dict, optional
    :param convergence: The convergence record, defaults to None.
    :type convergence: FemticConvergence, optional
    """

    def __init__(self, root_dir: PathLike, data: Optional[FemticData] = None,
                results: Optional[dict] = None, convergence: Optional[FemticConvergence] = None,
                name_dict: Optional[dict] = None):
        self.root_dir = pathlib.Path(root_dir)
        self.data = data
        self.results = dict(results) if results else {}
        self.convergence = convergence
        self.logger = logger
        # Mapping ``station_id -> display_name`` used by every plot
        # produced through this inversion. Populate via
        # :meth:`attach_names`, :meth:`attach_names_from_modem`, or
        # :meth:`attach_names_from_h5`.
        self.name_dict: dict = dict(name_dict) if name_dict else {}
        self._propagate_names()

    def _propagate_names(self) -> None:
        """Copy ``self.name_dict`` into every loaded child object."""
        for r in self.results.values():
            r.name_dict = self.name_dict

    def attach_names(self, source) -> dict:
        """Populate :attr:`name_dict` from one of several sources.

        :param source: One of

            - ``dict``: explicit ``{station_id: name}`` mapping (used
              verbatim).
            - ``list`` or ``tuple``: ordered list of names; assumed to
              correspond to station ids ``1..N`` in order.
            - ``pathlib.Path`` or ``str`` ending in ``.data``: a ModEM
              data file; names matched to FEMTIC stations by position.
            - ``pathlib.Path`` or ``str`` ending in ``.h5``: an mtpy-v2
              HDF5 collection; names matched by position (requires
              mtpy).
            - :class:`FemticData` (e.g. one loaded via
              :meth:`FemticData.from_modem`): names taken from its
              ``owner`` column and matched by position.

        :type source: dict, list, str, pathlib.Path, or FemticData

        :return: The resulting ``name_dict`` (also stored on the
            inversion).
        :rtype: dict

        :raises ValueError: If ``source`` cannot be interpreted.
        """
        if isinstance(source, dict):
            self.name_dict = {int(k) if isinstance(k, (int, np.integer))
                              or (isinstance(k, str) and k.isdigit()) else k:
                              str(v) for k, v in source.items()}
        elif isinstance(source, (list, tuple)):
            self.name_dict = {i + 1: str(name) for i, name in enumerate(source)}
        elif isinstance(source, FemticData):
            self.name_dict = self._match_to_femtic_data(source)
        else:
            path = pathlib.Path(source)
            if path.suffix.lower() == ".data":
                self.name_dict = self._match_to_femtic_data(
                    FemticData.from_modem(path))
            elif path.suffix.lower() == ".h5":
                ex = _read_h5_station_dataframe(path)
                if self.data is None:
                    raise ValueError(
                        "Cannot match by position: no FemticData loaded "
                        "in this FemticInversion."
                    )
                self.name_dict = _match_stations_by_position(
                    self.data.station_coords, ex,
                    code_col="Code", ex_x_col="X", ex_y_col="Y"
                )
            else:
                raise ValueError(
                    f"Unrecognized source for attach_names: {source!r}. "
                    f"Use a dict, list, .data, .h5, or FemticData."
                )
        self._propagate_names()
        return self.name_dict

    def _match_to_femtic_data(self, external: FemticData) -> dict:
        """Match this inversion's FEMTIC stations to ``external`` by position."""
        if self.data is None:
            raise ValueError(
                "Cannot match by position: no FemticData loaded in "
                "this FemticInversion."
            )
        ex_df = external.to_dataframe()
        if "owner" not in ex_df.columns or ex_df["owner"].isna().all():
            raise ValueError(
                "External FemticData has no 'owner' column with station "
                "names. Use a dict-based attach_names instead."
            )
        ex_coords = (ex_df[["owner", "X", "Y"]]
                     .drop_duplicates(subset=["owner"])
                     .rename(columns={"owner": "Code"}))
        return _match_stations_by_position(
            self.data.station_coords, ex_coords,
            code_col="Code", ex_x_col="X", ex_y_col="Y"
        )

    def attach_names_from_modem(self, modem_path: PathLike) -> dict:
        """Build :attr:`name_dict` from a ModEM ``.data`` file (matched by position).

        :param modem_path: Path to the ModEM data file.
        :type modem_path: str or pathlib.Path

        :return: The resulting ``name_dict``.
        :rtype: dict
        """
        return self.attach_names(pathlib.Path(modem_path))

    def attach_names_from_h5(self, h5_path: PathLike) -> dict:
        """Build :attr:`name_dict` from an mtpy-v2 ``.h5`` collection.

        :param h5_path: Path to the HDF5 collection.
        :type h5_path: str or pathlib.Path

        :return: The resulting ``name_dict``.
        :rtype: dict

        :raises ImportError: If mtpy is not installed.
        """
        return self.attach_names(pathlib.Path(h5_path))

    @classmethod
    def from_directory( cls, root_dir: PathLike, observe_filename: str = "observe.dat",
                        cnv_filename: str = "femtic.cnv", iterations: Optional[Iterable[int]] = None) -> "FemticInversion":
        """Discover and load all data in an inversion directory.

        Result CSVs are found whether they sit directly in ``root_dir``
        (flat layout) or inside per-iteration subdirectories named
        ``<root_dir>/iter_<N>/``. The ``observe.dat`` and ``femtic.cnv``
        files are read from ``root_dir`` itself.

        :param root_dir: The inversion directory.
        :type root_dir: str or pathlib.Path
        :param observe_filename: Filename of the observation file,
            defaults to ``"observe.dat"``.
        :type observe_filename: str, optional
        :param cnv_filename: Filename of the convergence file, defaults
            to ``"femtic.cnv"``.
        :type cnv_filename: str, optional
        :param iterations: Iterations to load. If None (default), every
            iteration with at least one ``result_*_iter*.csv`` is
            loaded.
        :type iterations: iterable[int], optional

        :return: A populated :class:`FemticInversion`.
        :rtype: FemticInversion
        """
        root = pathlib.Path(root_dir)
        data = None
        cnv = None

        # observe.dat and femtic.cnv live in the inversion root and are
        # identical across iterations, so they are read straight from there.
        obs_path = root / observe_filename
        if obs_path.exists():
            data = FemticData.from_file(obs_path)
        cnv_path = root / cnv_filename
        if cnv_path.exists():
            cnv = FemticConvergence.from_file(cnv_path)

        # Discover iterations. Result CSVs may sit directly in <root> (flat
        # layout) or inside per-iteration subdirectories named
        # ``<root>/iter_<N>/result_*_iter<N>.csv``. Map each iteration number
        # to the directory that actually holds its CSVs (the root wins on ties).
        iter_dirs: dict = {}
        search_dirs = [root] + sorted(
            d for d in root.glob("iter_*") if d.is_dir())
        for d in search_dirs:
            for p in d.glob("result_*_iter*.csv"):
                m = re.search(r"_iter(\d+)\.csv$", p.name)
                if m:
                    iter_dirs.setdefault(int(m.group(1)), d)
        present = set(iter_dirs)
        if iterations is not None:
            present &= set(iterations)
        results = {}
        for it in sorted(present):
            try:
                results[it] = FemticResponses.from_directory(
                    iter_dirs[it], iteration=it)
                if data is not None:
                    results[it] = results[it].with_coords(data)
            except FileNotFoundError:
                continue
        return cls(root, data=data, results=results, convergence=cnv)

    @property
    def iterations(self) -> list:
        """Sorted list of iterations for which responses are loaded."""
        return sorted(self.results.keys())

    @property
    def best_iteration(self) -> Optional[int]:
        """Iteration with lowest RMS, per the convergence file.

        Returns the largest iteration with a loaded :class:`FemticResponses`
        when no convergence file is available.
        """
        if self.convergence is not None:
            return self.convergence.best_iteration
        return max(self.iterations) if self.iterations else None

    def rms_analysis(self, iteration: Optional[int] = None, **kwargs) -> RMSAnalysis:
        """Convenience wrapper for ``self.results[iteration].rms_analysis()``.

        :param iteration: Iteration to analyze. Defaults to
            :attr:`best_iteration`.
        :type iteration: int, optional

        :return: A new :class:`RMSAnalysis`.
        :rtype: RMSAnalysis

        :raises KeyError: If the requested iteration was not loaded.
        """
        if iteration is None:
            iteration = self.best_iteration
        if iteration not in self.results:
            raise KeyError(
                f"Iteration {iteration} not loaded. "
                f"Available: {self.iterations}"
            )
        return self.results[iteration].rms_analysis(**kwargs)

    def __repr__(self) -> str:
        n_iters = len(self.iterations)
        n_stations = self.data.n_stations if self.data is not None else 0
        return (f"FemticInversion(root={self.root_dir.name!r}, "
                f"n_stations={n_stations}, n_iters={n_iters}, "
                f"best_iter={self.best_iteration})")



# observe.dat writing from an mtpy-v2 long DataFrame
# 
# The functions below operate on the *long* DataFrame produced by 
# ``mtpy.MTData.to_dataframe()`` (one row per station/period, with complex 
# ``z_*`` columns and ``z_*_error`` columns, plus optional ``res_*``/
# ``phase_*`` apparent-resistivity columns and ``t_z*`` tipper columns). 
# This is a different path from :meth:`FemticData.to_file`, which writes 
# from this module's *wide* scheme.
# 
# FEMTIC station-coordinate convention (matching makeDHexaMesh and
# ``hexmesh.write_obs_site``): X = north written first, Y = east
# written second, both in km.


def convert_units(data_orig: pd.DataFrame, data_type="Z") -> pd.DataFrame:
    """Convert impedance columns from ``[mV/km]/[nT]`` to ohm.

    FEMTIC works in ohm, while mtpy/ModEM use ``[mV/km]/[nT]``. The
    conversion divides every impedance and impedance-error column by
    :data:`OHM_TO_MVKMNT` (``10000 / (4 \\pi)``).

    :param data_orig: An mtpy-v2 long DataFrame with ``z_xx`` … ``z_yy``
        and ``z_xx_error`` … ``z_yy_error`` columns.
    :type data_orig: pandas.DataFrame
    :param data_type: Which data families to convert. Only the ``"Z"``
        family is rescaled; anything else is passed through unchanged.
        Accepts a string or a list/tuple of strings. Defaults to ``"Z"``.
    :type data_type: str or list, optional

    :return: A converted copy; the input is not mutated.
    :rtype: pandas.DataFrame
    """
    C = OHM_TO_MVKMNT
    data = copy.deepcopy(data_orig)
    if "Z" in data_type:
        z_cols = ["z_xx", "z_xy", "z_yx", "z_yy"]
        err_cols = ["z_xx_error", "z_xy_error", "z_yx_error", "z_yy_error"]
        data.loc[:, z_cols] = data_orig.loc[:, z_cols] / C
        data.loc[:, err_cols] = data_orig.loc[:, err_cols] / C
    return data


def conjugate(data_orig: pd.DataFrame, data_type) -> pd.DataFrame:
    """Take the complex conjugate of each impedance (and tipper) component.

    FEMTIC stores responses under the ``exp(-i \\omega t)`` convention,
    whereas mtpy uses ``exp(+i \\omega t)``; conjugating the complex
    components on the way out flips between the two.

    :param data_orig: An mtpy-v2 long DataFrame.
    :type data_orig: pandas.DataFrame
    :param data_type: Data families to conjugate. ``"Z"`` conjugates the
        four impedance components; ``"VTF"`` conjugates ``t_zx`` and
        ``t_zy``. Accepts a string or list of strings.
    :type data_type: str or list

    :return: A conjugated copy; the input is not mutated.
    :rtype: pandas.DataFrame
    """
    data = copy.deepcopy(data_orig)
    if "Z" in data_type:
        data.loc[:, ["z_xx", "z_xy", "z_yx", "z_yy"]] = \
            data_orig.loc[:, ["z_xx", "z_xy", "z_yx", "z_yy"]].apply(np.conj)
    if "VTF" in data_type:
        data.loc[:, ["t_zx", "t_zy"]] = \
            data_orig.loc[:, ["t_zx", "t_zy"]].apply(np.conj)
    return data


def apply_error_floor(data: pd.DataFrame, data_type,
                      error_floor_Z: float = 0.05,
                      error_floor_AR: float = 0.05,
                      error_floor_Tz: float = 0.05) -> pd.DataFrame:
    """Impose a minimum standard deviation (error floor) on each datum.

    For impedance (``"Z"``), the floor for a given station/period is the
    geometric mean of the off-diagonal magnitudes,
    ``sqrt(|Zxy| * |Zyx|)``, scaled by ``error_floor_Z``; it is applied to
    the real and imaginary error of all four components. Eight new
    columns are added — ``z_xx_R_error``, ``z_xx_I_error``, …,
    ``z_yy_I_error`` — which :func:`write_mtdata` then writes out.

    For apparent resistivity / phase (``"AR"``) and tipper (``"VTF"``) the
    relevant ``*_error`` columns are floored in place.

    :param data: An mtpy-v2 long DataFrame. Modified in place *and*
        returned (per the original behaviour).
    :type data: pandas.DataFrame
    :param data_type: Iterable/str of families to floor: any of ``"Z"``,
        ``"AR"``, ``"VTF"``.
    :type data_type: str or list
    :param error_floor_Z: Impedance error floor as a fraction (e.g.
        ``0.05`` for 5%), defaults to ``0.05``.
    :type error_floor_Z: float, optional
    :param error_floor_AR: Apparent-resistivity error floor as a fraction,
        defaults to ``0.05``.
    :type error_floor_AR: float, optional
    :param error_floor_Tz: Tipper error floor (absolute), defaults to
        ``0.05``.
    :type error_floor_Tz: float, optional

    :return: The floored DataFrame (same object that was passed in).
    :rtype: pandas.DataFrame
    """
    if "Z" in data_type:
        # Split each component's single error into separate real/imag
        # error columns (FEMTIC writes Re and Im errors independently).
        for comp in ("z_xx", "z_xy", "z_yx", "z_yy"):
            data[f"{comp}_R_error"] = copy.deepcopy(data[f"{comp}_error"])
            data[f"{comp}_I_error"] = copy.deepcopy(data[f"{comp}_error"])

        error_cols = [
            "z_xx_R_error", "z_xx_I_error", "z_xy_R_error", "z_xy_I_error",
            "z_yx_R_error", "z_yx_I_error", "z_yy_R_error", "z_yy_I_error",
        ]
        for each in range(len(data)):
            # Geometric mean of the off-diagonal magnitudes, scaled by the
            # requested floor fraction.
            error_floor = np.sqrt(
                np.abs(data["z_xy"].iloc[each]) * np.abs(data["z_yx"].iloc[each])
            ) * np.array(error_floor_Z)
            row = data.index[each]
            for comp in error_cols:
                if np.abs(data.loc[row, comp]) < error_floor:
                    data.loc[row, comp] = error_floor

    if "AR" in data_type:
        std = data[["res_xx_error", "res_xy_error", "res_yx_error", "res_yy_error"]]
        valkey = ["res_xx", "res_xy", "res_yx", "res_yy"]
        for each in range(len(data)):
            for i, comp in enumerate(["res_xx_error", "res_xy_error",
                                      "res_yx_error", "res_yy_error"]):
                error_floor = error_floor_AR * list(data[valkey[i]])[each]
                if list(std[comp])[each] < error_floor:
                    data.loc[data.index[each], comp] = error_floor

        std = data[["phase_xx_error", "phase_xy_error",
                    "phase_yx_error", "phase_yy_error"]]
        for each in range(len(data)):
            for comp in ("phase_xx_error", "phase_xy_error",
                         "phase_yx_error", "phase_yy_error"):
                # 0.572958 deg ≈ 1% in phase; scale by the AR floor fraction.
                error_floor = error_floor_AR * 0.572958 * 100
                if list(std[comp])[each] < error_floor:
                    data.loc[data.index[each], comp] = error_floor

    if "VTF" in data_type:
        std = data[["t_zx_error", "t_zy_error"]]
        for each in range(len(data)):
            for comp in ("t_zx_error", "t_zy_error"):
                if std[comp][each] < error_floor_Tz:
                    data.loc[data.index[each], comp] = error_floor_Tz

    return data


def drop_nodata(data_orig: pd.DataFrame, data_type) -> pd.DataFrame:
    """Drop any station/period row carrying a missing-data value.

    This mirrors the nodata screening in ``hseille/femticPy``'s
    ``read_MTdata``: rather than masking individual tensor components, the
    *entire* record for a station at a given frequency is removed if any
    relevant component is missing. For impedance (``"Z"``) the screened
    columns are the four complex ``z_*`` values together with their
    ``z_*_error`` standard deviations; for apparent resistivity / phase
    (``"AR"``) they are the ``res_*``/``phase_*`` values and their errors;
    for tipper (``"VTF"``) they are ``t_zx``/``t_zy`` and their errors.

    A value counts as missing if it is NaN (the
    ``>= 1e10`` magnitude threshold femticPy uses on the EDI ``1.0e+32``
    EMPTY flag silently passes literal NaN through, so NaN is handled
    explicitly here) or if it matches the FEMTIC :data:`NODATA_VAL`
    sentinel.

    :param data_orig: An mtpy-v2 long DataFrame.
    :type data_orig: pandas.DataFrame
    :param data_type: Families to screen: any of ``"Z"``, ``"AR"``,
        ``"VTF"`` (string or list).
    :type data_type: str or list

    :return: A screened copy with offending rows removed; the input is not
        mutated.
    :rtype: pandas.DataFrame
    """
    data = copy.deepcopy(data_orig)

    screen_cols: list = []
    if "Z" in data_type:
        screen_cols += [
            "z_xx", "z_xy", "z_yx", "z_yy",
            "z_xx_error", "z_xy_error", "z_yx_error", "z_yy_error",
        ]
    if "AR" in data_type:
        screen_cols += [
            "res_xx", "res_xy", "res_yx", "res_yy",
            "phase_xx", "phase_xy", "phase_yx", "phase_yy",
            "res_xx_error", "res_xy_error", "res_yx_error", "res_yy_error",
            "phase_xx_error", "phase_xy_error",
            "phase_yx_error", "phase_yy_error",
        ]
    if "VTF" in data_type:
        screen_cols += ["t_zx", "t_zy", "t_zx_error", "t_zy_error"]

    screen_cols = [c for c in screen_cols if c in data.columns]
    if not screen_cols:
        return data

    block = data[screen_cols]
    bad = block.isna().any(axis=1)
    bad |= (block == NODATA_VAL).any(axis=1)

    n_dropped = int(bad.sum())
    if n_dropped:
        logger.info(
            f"drop_nodata: removing {n_dropped} station/period record(s) "
            f"with missing data across {screen_cols}."
        )
    return data.loc[~bad].reset_index(drop=True)


def prep_data(data_orig: pd.DataFrame, data_type,
              error_floor_Z: float = 0.05,
              error_floor_AR: float = 0.05,
              error_floor_Tz: float = 0.05) -> pd.DataFrame:
    """Run the full FEMTIC prep chain on an mtpy-v2 long DataFrame.

    Equivalent to :func:`convert_units` -> :func:`conjugate` ->
    :func:`apply_error_floor`. The result is in ohm, under FEMTIC's
    ``exp(-i \\omega t)`` convention, with error floors applied.

    :param data_orig: An mtpy-v2 long DataFrame.
    :type data_orig: pandas.DataFrame
    :param data_type: Families to prepare: any of ``"Z"``, ``"AR"``,
        ``"VTF"`` (string or list).
    :type data_type: str or list
    :param error_floor_Z: Impedance error floor fraction, defaults to ``0.05``.
    :type error_floor_Z: float, optional
    :param error_floor_AR: Apparent-resistivity error floor fraction,
        defaults to ``0.05``.
    :type error_floor_AR: float, optional
    :param error_floor_Tz: Tipper error floor, defaults to ``0.05``.
    :type error_floor_Tz: float, optional

    :return: The prepared DataFrame.
    :rtype: pandas.DataFrame
    """
    data = copy.deepcopy(data_orig)
    data = drop_nodata(data, data_type)
    data = convert_units(data, data_type)
    data = conjugate(data, data_type)
    data = apply_error_floor(data, data_type, error_floor_Z,
                             error_floor_AR, error_floor_Tz)
    return data


def write_mtdata(mt_df: pd.DataFrame,
                 outdir: Optional[PathLike] = None,
                 invert_Z: bool = True, error_floor_Z: float = 0.05,
                 invert_AppRes: bool = False, error_floor_AR: float = 0.05,
                 invert_T: bool = False, error_floor_Tz: float = 0.05) -> None:
    """Write a FEMTIC ``observe.dat`` from an mtpy-v2 long DataFrame.

    The selected data families are unit-converted, conjugated, and
    error-floored via :func:`prep_data`, then written as the requested
    blocks: ``MT`` (impedance), ``APP_RES_AND_PHS``, and/or ``VTF``. The
    file is always named ``observe.dat`` inside ``outdir``.

    Station coordinates are written as ``X`` (north) then ``Y`` (east) in
    km, matching FEMTIC's convention and ``hexmesh``. Station
    IDs are offset per block so the families do not collide: impedance
    uses ``1, 2, …``; apparent resistivity uses ``101, 102, …``; tipper
    uses ``1001, 1002, …``.

    :param mt_df: mtpy-v2 long DataFrame (``mt_data.to_dataframe()``) with
        ``station``, ``period``, ``east``, ``north`` and the complex
        ``z_*`` / ``z_*_error`` columns (and ``res_*``/``phase_*`` or
        ``t_z*`` columns if writing those blocks).
    :type mt_df: pandas.DataFrame
    :param outdir: Directory to write ``observe.dat`` into. Defaults to
        the current working directory.
    :type outdir: str or pathlib.Path, optional
    :param invert_Z: Write the impedance (``MT``) block, defaults to True.
    :type invert_Z: bool, optional
    :param error_floor_Z: Impedance error floor fraction, defaults to ``0.05``.
    :type error_floor_Z: float, optional
    :param invert_AppRes: Write the apparent-resistivity/phase block,
        defaults to False.
    :type invert_AppRes: bool, optional
    :param error_floor_AR: Apparent-resistivity error floor fraction,
        defaults to ``0.05``.
    :type error_floor_AR: float, optional
    :param invert_T: Write the tipper (``VTF``) block, defaults to False.
    :type invert_T: bool, optional
    :param error_floor_Tz: Tipper error floor, defaults to ``0.05``.
    :type error_floor_Tz: float, optional

    :return: ``None``. Writes ``<outdir>/observe.dat`` as a side effect.
    :rtype: None
    """
    outdir = pathlib.Path.cwd() if outdir is None else pathlib.Path(outdir)

    data_type = []
    if invert_Z:
        data_type.append("Z")
    if invert_AppRes:
        data_type.append("AR")
    if invert_T:
        data_type.append("VTF")

    mt_dataframe = prep_data(mt_df, data_type, error_floor_Z=error_floor_Z,
                             error_floor_AR=error_floor_AR,
                             error_floor_Tz=error_floor_Tz)

    with open(outdir.joinpath("observe.dat"), "w") as file:
        if invert_Z:
            file.write(f"MT    {len(mt_dataframe['station'].unique()):d}\n")
            for i, s in enumerate(mt_dataframe["station"].unique()):
                station = (mt_dataframe[mt_dataframe["station"] == s]
                           .sort_values(by=["period"], ascending=True)
                           .reset_index())
                file.write(f"{i+1}  {i+1}  "
                           f"{float(station['north'][0]/1000.0):.3f}  "
                           f"{float(station['east'][0]/1000.0):.3f}  \n")
                file.write(f"{len(station['period'])}\n")
                for p in station["period"]:
                    period = station[station["period"] == p]
                    try:
                        file.write(
                            f"{float(1.0/period['period'].item()):.5f} "
                            f"{float(np.squeeze(np.real(period['z_xx']))):.4e} {float(np.squeeze(np.imag(period['z_xx']))):.4e} "
                            f"{float(np.squeeze(np.real(period['z_xy']))):.4e} {float(np.squeeze(np.imag(period['z_xy']))):.4e} "
                            f"{float(np.squeeze(np.real(period['z_yx']))):.4e} {float(np.squeeze(np.imag(period['z_yx']))):.4e} "
                            f"{float(np.squeeze(np.real(period['z_yy']))):.4e} {float(np.squeeze(np.imag(period['z_yy']))):.4e} "
                            f"{float(np.squeeze(period['z_xx_R_error'])):.4e} {float(np.squeeze(period['z_xx_I_error'])):.4e} "
                            f"{float(np.squeeze(period['z_xy_R_error'])):.4e} {float(np.squeeze(period['z_xy_I_error'])):.4e} "
                            f"{float(np.squeeze(period['z_yx_R_error'])):.4e} {float(np.squeeze(period['z_yx_I_error'])):.4e} "
                            f"{float(np.squeeze(period['z_yy_R_error'])):.4e} {float(np.squeeze(period['z_yy_I_error'])):.4e} \n")
                    except Exception as e:
                        logger.debug(f"Failed writing period: {period['period']}")
                        logger.debug(f"Offending row: {period}")
                        raise e

        if invert_T:
            file.write(f"VTF    {len(mt_dataframe['station'].unique()):d}\n")
            for i, s in enumerate(mt_dataframe["station"].unique()):
                station = (mt_dataframe[mt_dataframe["station"] == s]
                           .sort_values(by=["period"], ascending=True)
                           .reset_index())
                file.write(f"{i+1001}  {i+1001}  "
                           f"{float(station['north'][0]/1000.0):.3f}  "
                           f"{float(station['east'][0]/1000.0):.3f}  \n")
                file.write(f"{len(station['period'])}\n")
                for p in station["period"]:
                    period = station[station["period"] == p]
                    file.write(
                        f"{float(1.0/period['period'].item()):.5f} "
                        f"{float(np.squeeze(np.real(period['t_zx']))):.4e} {float(np.squeeze(np.imag(period['t_zx']))):.4e} "
                        f"{float(np.squeeze(np.real(period['t_zy']))):.4e} {float(np.squeeze(np.imag(period['t_zy']))):.4e} "
                        f"{float(np.squeeze(period['t_zx_error'])):.4e} {float(np.squeeze(period['t_zx_error'])):.4e} "
                        f"{float(np.squeeze(period['t_zy_error'])):.4e} {float(np.squeeze(period['t_zy_error'])):.4e} \n")

        if invert_AppRes:
            file.write(f"APP_RES_AND_PHS {len(mt_dataframe['station'].unique()):d}\n")
            for i, s in enumerate(mt_dataframe["station"].unique()):
                station = (mt_dataframe[mt_dataframe["station"] == s]
                           .sort_values(by=["period"], ascending=False)
                           .reset_index())
                file.write(f"{i+101}  {i+101}  "
                           f"{float(station['east'][0]/1000.0):>8.6f}  "
                           f"{float(station['north'][0]/1000.0):>8.6f}  \n")
                file.write(f"{len(station['period'])}\n")
                for p in station["period"]:
                    period = station[station["period"] == p]
                    file.write(
                        f"{float(1.0/period['period'].item()):.5f} "
                        f"{float(np.squeeze(period['res_xx'])):.4e} {float(np.squeeze(period['phase_xx'])):.4e} "
                        f"{float(np.squeeze(period['res_xy'])):.4e} {float(np.squeeze(period['phase_xy'])):.4e} "
                        f"{float(np.squeeze(period['res_yx'])):.4e} {float(np.squeeze(period['phase_yx'])):.4e} "
                        f"{float(np.squeeze(period['res_yy'])):.4e} {float(np.squeeze(period['phase_yy'])):.4e} "
                        f"{float(np.squeeze(period['res_xx_error'])):.4e} {-float(np.squeeze(period['phase_xx_error'])):.4e} "
                        f"{float(np.squeeze(period['res_xy_error'])):.4e} {float(np.squeeze(period['phase_xy_error'])):.4e} "
                        f"{float(np.squeeze(period['res_yx_error'])):.4e} {float(np.squeeze(period['phase_yx_error'])):.4e} "
                        f"{float(np.squeeze(period['res_yy_error'])):.4e} {-float(np.squeeze(period['phase_yy_error'])):.4e} \n")

        file.write("END")
        logger.info("observe.dat written")
    return

