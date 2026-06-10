"""ModEM bridges for the FEMTIC toolkit.

You'll probably want to exclude this from a full, merged version.

Two experimental converters live here:

1. **MT impedance data** (ModEM ``.data`` <-> FEMTIC ``observe.dat``).
   These are thin wrappers around :class:`responses.FemticData`,
   which is the single canonical implementation of the parsing, the
   ``X = north`` / ``Y = east`` axis swap, the unit conversion
   (:data:`responses.OHM_TO_MVKMNT`) and the ``exp(+/- i w t)``
   sign handling. Use :class:`responses.FemticData` directly for
   anything beyond the one-shot file conversions offered here.

2. **ModEM resistivity model** (``.rho`` / ``.model`` rectilinear files):
   :class:`ModEMModel` reads/writes the model file, exposes the grid, and
   provides the ModEM->FEMTIC mesh helpers plus optional pyvista exports
   (3-D grid, depth slices, VTK, ``.cov`` writer). pyvista is an optional
   dependency: it is detected once at import, model-file I/O works without
   it, and the pyvista-backed functions raise a clear :class:`ImportError`
   before doing any work if it is missing.

ModEM model-frame axes: ``X = north``, ``Y = east``, ``Z`` positive down.

@author: oaazeved

"""

from __future__ import annotations

import pathlib
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .responses import FemticData, PathLike

__all__ = [
    "write_modem_to_femtic", "write_femtic_to_modem",
    "ModEMModel", "modem_data_to_femtic_obssite",
    "array_to_rgba", "write_slices_to_model_file", "write_slices_to_cov_file",
]



# Optional pyvista dependency

# pyvista is only needed for the 3-D grid / slice / VTK / .cov exports. It is
# detected once at import so the rest of the module (model-file and data I/O)
# works whether or not it is installed; the pyvista-backed functions raise a
# clear error before doing any work if it is missing.
try:
    import pyvista as _pv
    _HAVE_PYVISTA = True
except ImportError:  # pragma: no cover
    _pv = None
    _HAVE_PYVISTA = False


def _require_pyvista():
    """Return the imported pyvista module, or raise a clear error if absent.

    :return: The ``pyvista`` module.
    :raises ImportError: If pyvista is not installed in the current
        environment, with guidance on how to enable these features.
    """
    if not _HAVE_PYVISTA:
        raise ImportError(
            "pyvista is required for this function but is not installed in "
            "the current environment. Install it (e.g. `pip install pyvista`) "
            "to use the ModEMModel grid/slice/VTK exports and the slice "
            "writers. All other femtic_modem functionality works without it."
        )
    return _pv


# ===========================================================================
# MT impedance data: ModEM <-> FEMTIC  (delegated to FemticData)
# ===========================================================================

def write_modem_to_femtic(modem_data_path: PathLike,
                          femtic_observe_path: PathLike) -> FemticData:
    """Convert a ModEM impedance ``.data`` file to a FEMTIC ``observe.dat``.

    Equivalent to ``FemticData.from_modem(src).to_file(dst)``. Units and
    sign convention are taken from the ModEM file header; the output is in
    FEMTIC's ohm / ``exp(-i w t)`` convention.

    :param modem_data_path: Source ModEM data file.
    :type modem_data_path: str or pathlib.Path
    :param femtic_observe_path: Destination FEMTIC observation file.
    :type femtic_observe_path: str or pathlib.Path
    :return: The intermediate :class:`FemticData` (in case the caller wants
        to inspect it).
    :rtype: responses.FemticData
    """
    data = FemticData.from_modem(modem_data_path)
    data.to_file(femtic_observe_path)
    return data


def write_femtic_to_modem(femtic_observe_path: PathLike,
                          modem_data_path: PathLike) -> FemticData:
    """Convert a FEMTIC ``observe.dat`` to a ModEM impedance ``.data`` file.

    Equivalent to ``FemticData.from_file(src).to_modem(dst)``. The ModEM
    output is in ``[mV/km]/[nT]`` and ``exp(+i w t)``.

    :param femtic_observe_path: Source FEMTIC observation file.
    :type femtic_observe_path: str or pathlib.Path
    :param modem_data_path: Destination ModEM data file.
    :type modem_data_path: str or pathlib.Path
    :return: The intermediate :class:`FemticData`.
    :rtype: responses.FemticData
    """
    data = FemticData.from_file(femtic_observe_path)
    data.to_modem(modem_data_path)
    return data


def modem_data_to_femtic_obssite(modem_df: pd.DataFrame, write_path: PathLike,
                                 cubes: Optional[list] = None) -> None:
    """Write a FEMTIC ``obs_site.dat`` from a ModEM-style station DataFrame.

    :param modem_df: DataFrame with ``Station``, ``X``, ``Y``, ``Z`` columns
        (X/Y in meters, ModEM frame).
    :type modem_df: pandas.DataFrame
    :param write_path: Output path.
    :type write_path: str or pathlib.Path
    :param cubes: Optional list of ``(radius_km, len_km, factor)`` refinement
        cubes appended to each station line, defaults to ``None``.
    :type cubes: list or None, optional
    """
    cubes = cubes or []
    with open(write_path, "w") as f:
        f.write(f"{len(modem_df['Station'].unique())}\n")
        for sta in modem_df["Station"].unique():
            station = modem_df.loc[modem_df["Station"] == sta]
            f.write(f"{station['X'].values[0] / 1000:.4f} "
                    f"{station['Y'].values[0] / 1000:.4f} "
                    f"{station['Z'].values[0]:.1f}   {len(cubes)} ")
            for cube in cubes:
                if cube is not None and len(cube) > 0:
                    f.write(f" {cube[0]:.1f}  {cube[1]:.1f} {cube[2]:.1f}")
            f.write("\n")


# ===========================================================================
# ModEM resistivity model (.rho / .model)
# ===========================================================================

def _read_modem_model_file(filepath) -> dict:
    """Parse a ModEM rectilinear model file into a dict.

    Robust to the header carrying either 4 tokens (``Nx Ny Nz Nair``) or 5
    (``Nx Ny Nz Nair valtype``; a missing ``valtype`` defaults to
    ``"LINEAR"``), and to vectors/value-blocks that are **wrapped across
    multiple physical lines** (as in the ModEM example data) — everything
    after the header is consumed as a single whitespace-separated token
    stream: ``Nx`` x-spacings, ``Ny`` y-spacings, ``Nz`` z-spacings, then
    ``Nx*Ny*Nz`` resistivity values, then 3 origin values and the rotation.

    The resistivity ordering matches :func:`_write_modem_model_file`
    (``x`` fastest, then ``y``, then ``z``), so write->read round-trips
    exactly.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    lines = [ln for ln in lines if not ln.strip().startswith('#')]

    header = lines.pop(0).strip().split()
    if len(header) >= 5:
        cellXnum, cellYnum, cellZnum, cellAirnum, valtype = header[:5]
    elif len(header) == 4:
        cellXnum, cellYnum, cellZnum, cellAirnum = header[:4]
        valtype = "LINEAR"
    else:
        raise ValueError(f"Unexpected model header: {header!r}")
    nx, ny, nz, cellAirnum = (int(cellXnum), int(cellYnum),
                              int(cellZnum), int(cellAirnum))

    # Flat token stream for everything after the header (handles wrapping).
    toks = " ".join(lines).split()
    vals = np.array(toks, dtype=float)

    i = 0
    dX = vals[i:i + nx]; i += nx
    dY = vals[i:i + ny]; i += ny
    dZ = vals[i:i + nz]; i += nz
    n_cells = nx * ny * nz
    # x fastest, then y, then z  (transpose of a (nz, ny, nx) reshape).
    data = vals[i:i + n_cells].reshape(nz, ny, nx).transpose(2, 1, 0).copy()
    i += n_cells
    origin = list(vals[i:i + 3]) if i + 3 <= len(vals) else [0.0, 0.0, 0.0]
    i += 3
    rot_angle = float(vals[i]) if i < len(vals) else 0.0

    X = np.insert(np.cumsum(dX), 0, 0) + origin[0]
    Y = np.insert(np.cumsum(dY), 0, 0) + origin[1]
    Z = np.insert(np.cumsum(dZ), 0, 0) + origin[2]

    return {
        'cellXnum': nx, 'cellYnum': ny, 'cellZnum': nz,
        'cellAirnum': cellAirnum, 'valtype': valtype,
        'dX': dX, 'dY': dY, 'dZ': dZ, 'X': X, 'Y': Y, 'Z': Z,
        'data': data, 'origin': origin, 'rot_angle': rot_angle,
    }


def _write_modem_model_file(model_dict, filepath) -> None:
    """Write a ModEM rectilinear model file."""
    with open(filepath, 'w') as f:
        f.write("# MODEL FILE WRITTEN BY mtpy.modeling.femtic.modem.ModEMModel\n")
        f.write(f"  {model_dict['cellXnum']}  {model_dict['cellYnum']}  "
                f"{model_dict['cellZnum']}  {model_dict['cellAirnum']} "
                f"{model_dict['valtype']}\n")
        f.write(f"{''.join([f'{abs(n):>12.3f}' for n in model_dict['dX']])}\n")
        f.write(f"{''.join([f'{abs(n):>12.3f}' for n in model_dict['dY']])}\n")
        f.write(f"{''.join([f'{abs(n):>12.3f}' for n in model_dict['dZ']])}\n")
        d_lines = []
        for zz in range(len(model_dict['dZ'])):
            d_lines.append("")
            for ee in range(len(model_dict['dY'])):
                line = []
                for nn in range(len(model_dict['dX'])):
                    line.append(f"{model_dict['data'][nn, ee, zz]:>13.5E}")
                d_lines.append("".join(line))
        f.write('\n'.join(d_lines))
        f.write(f"\n{' '.join([f'{c}' for c in model_dict['origin']])}\n")
        f.write(f"{model_dict['rot_angle']}")
    return None


class ModEMModel:
    """A ModEM rectilinear resistivity model.

    Wraps the parsed model dict (cell counts, spacings ``dX``/``dY``/``dZ``,
    edge coordinates ``X``/``Y``/``Z``, ``origin``, ``rot_angle``, and the
    ``data`` cube) and provides ModEM->FEMTIC helpers plus optional pyvista
    exports. Axes are the ModEM convention: ``X = north``, ``Y = east``,
    ``Z`` positive down.

    :param model_dict: Parsed model dictionary (use :meth:`from_file`).
    :type model_dict: dict
    """

    def __init__(self, model_dict: dict):
        self.model = model_dict
        self.logger = logger

    @classmethod
    def from_file(cls, filepath: PathLike) -> "ModEMModel":
        """Read a ModEM model file.

        :param filepath: Path to the ``.rho`` / ``.model`` file.
        :type filepath: str or pathlib.Path
        :return: A :class:`ModEMModel`.
        :rtype: ModEMModel
        """
        return cls(_read_modem_model_file(filepath))

    def to_file(self, filepath: PathLike) -> None:
        """Write the model back out in ModEM rectilinear format.

        :param filepath: Destination path.
        :type filepath: str or pathlib.Path
        """
        _write_modem_model_file(self.model, filepath)
        self.logger.info(f"Wrote ModEM model file to: {filepath}")

    # accessors 
    @property
    def shape(self) -> tuple:
        """``(nx, ny, nz)`` cell counts."""
        m = self.model
        return (m['cellXnum'], m['cellYnum'], m['cellZnum'])

    @property
    def n_air(self) -> int:
        """Number of air layers recorded in the header."""
        return self.model['cellAirnum']

    @property
    def valtype(self) -> str:
        """Value type string (``LOGE``, ``LOG10``, ``LINEAR``)."""
        return self.model['valtype']

    @property
    def origin(self) -> list:
        """Model origin ``[x0, y0, z0]`` (meters, ModEM frame)."""
        return self.model['origin']

    @property
    def rotation_deg(self) -> float:
        """Model rotation angle (degrees)."""
        return self.model['rot_angle']

    @property
    def data(self) -> np.ndarray:
        """The raw ``(nx, ny, nz)`` value cube (as stored, per ``valtype``)."""
        return self.model['data']

    def resistivity(self) -> np.ndarray:
        """Return the cube converted to linear ohm-m, honouring ``valtype``.

        :return: ``(nx, ny, nz)`` array of resistivities in ohm-m.
        :rtype: numpy.ndarray
        :raises NotImplementedError: For an unrecognised ``valtype``.
        """
        vt = self.model['valtype'].lower()
        d = self.model['data']
        if vt == 'loge':
            return np.exp(d)
        if vt == 'log10':
            return 10.0 ** d
        if vt == 'linear':
            return d
        raise NotImplementedError(f"Unknown valtype {self.model['valtype']!r}.")

    # FEMTIC mesh helpers ---------------------------------------------
    def to_femtic_meshgen(self, write_path: PathLike, init_res: float = 1000.0,
                          air_res: float = 1.0e10) -> None:
        """Write a FEMTIC ``meshgen.inp`` reusing this model's grid lines.

        :param write_path: Output path for ``meshgen.inp``.
        :type write_path: str or pathlib.Path
        :param init_res: Initial resistivity (ohm-m), defaults to ``1000.0``.
        :type init_res: float, optional
        :param air_res: Air resistivity (ohm-m), defaults to ``1e10``.
        :type air_res: float, optional
        """
        m = self.model
        with open(write_path, "w") as f:
            f.write("DIVISION_NUMBERS\n")
            f.write(f"{m['cellXnum']}, {m['cellYnum']}, {m['cellZnum']}\n")
            f.write("X_COORDINATES\n")
            for x in m['X']:
                f.write(f"{x / 1000:.3f}\n")
            f.write("Y_COORDINATES\n")
            for y in m['Y']:
                f.write(f"{y / 1000:.3f}\n")
            f.write("Z_COORDINATES\n")
            for z in m['Z']:
                f.write(f"{z / 1000:.3f}\n")
            f.write("INITIAL_RESISTIVITY\n")
            f.write(f"{init_res}\n")
            f.write("AIR_RESISTIVITY\n")
            f.write(f"{air_res:.1e}\n")
            f.write("END\n")
        self.logger.info(f"Wrote FEMTIC meshgen.inp to: {write_path}")

    # pyvista exports (lazy) ------------------------------------------
    def to_rectilinear_grid(self):
        """Build a pyvista structured grid coloured by linear resistivity.

        :return: A pyvista grid with ``Resistivity[Ohm-m]`` cell data.
        :raises ImportError: If pyvista is not installed.
        """
        pv = _require_pyvista()
        m = self.model
        x_edges = np.concatenate(([m['origin'][1]],
                                  np.cumsum(m['dY']) + m['origin'][1]))
        y_edges = np.concatenate(([m['origin'][0]],
                                  np.cumsum(m['dX']) + m['origin'][0]))
        z_edges = np.concatenate(([m['origin'][2]],
                                  np.cumsum(m['dZ']) + m['origin'][2]))
        grid = pv.RectilinearGrid(x_edges, y_edges, z_edges
                                  ).cast_to_structured_grid()
        vt = m['valtype'].lower()
        rot = np.rot90(m['data'])
        if vt == 'loge':
            grid.cell_data["Resistivity[Ohm-m]"] = np.exp(rot.ravel(order="F"))
        elif vt == 'log10':
            grid.cell_data["Resistivity[Ohm-m]"] = 10.0 ** rot.ravel(order="F")
        elif vt == 'linear':
            grid.cell_data["Resistivity[Ohm-m]"] = rot.ravel(order="F")
        else:
            raise NotImplementedError(
                f"Resistivity type {m['valtype']!r} not implemented.")
        return grid

    def to_vtk(self, write_path: PathLike) -> None:
        """Save the model as an ASCII VTK file (via pyvista)."""
        grid = self.to_rectilinear_grid()
        grid.save(str(write_path), binary=False)
        self.logger.info(f"Wrote VTK model file to: {write_path}")

    def z_slices(self, slice_depths) -> list:
        """Return horizontal (``-z``) slices of the grid at ``slice_depths``."""
        grid = self.to_rectilinear_grid()
        return get_pyvista_z_slices(grid, slice_depths)


# ===========================================================================
# pyvista slice / cov helpers (lazy)
# ===========================================================================

def array_to_rgba(input_array, colormap_name: str = 'plasma'):
    """Map an array to RGBA via a matplotlib colormap (NaN-masks values >= 39).

    :param input_array: Array of up to 3 dimensions.
    :param colormap_name: matplotlib colormap name, defaults to ``'plasma'``.
    :return: RGBA array with a trailing channel axis.
    :rtype: numpy.ndarray
    """
    import matplotlib.pyplot as plt
    input_array = np.asarray(input_array, dtype=float)
    mask = input_array >= 39
    input_array[mask] = np.nan
    norm_array = ((input_array - np.nanmin(input_array)) /
                  (np.nanmax(input_array) - np.nanmin(input_array)))
    colormap = plt.get_cmap(colormap_name)
    colormap.set_bad(color='w', alpha=0)
    return colormap(norm_array, bytes=False)


def get_pyvista_z_slices(grid, slice_depths) -> list:
    """Horizontal ``-z`` slices of ``grid`` at each depth in ``slice_depths``.

    :raises ImportError: If pyvista is not installed.
    """
    _require_pyvista()
    return [grid.slice(normal='-z', origin=[0, 0, d]) for d in slice_depths]


def write_slices_to_model_file(slices, field_name, output_path, model_dict=None,
                               fmt="%.5E", flipx=False, flipy=False) -> None:
    """Write pyvista horizontal slices into a ModEM model file.

    :param slices: List of pyvista slices ordered top->bottom.
    :param field_name: Cell-data array to write.
    :param output_path: Destination path.
    :param model_dict: Optional model dict supplying the grid header/spacings.
    :param fmt: Float format, defaults to ``"%.5E"``.
    :param flipx: Flip rows, defaults to ``False``.
    :param flipy: Flip columns, defaults to ``False``.
    :raises ImportError: If pyvista is not installed.
    """
    _require_pyvista()
    first_pts = slices[0].points
    ny = len(np.unique(first_pts[:, 1]))
    nx = len(np.unique(first_pts[:, 0]))
    nz = len(slices)
    lines = f"{ny} {nx} {nz}\n"
    if model_dict:
        lines = "# Written by write_slices_to_model_file\n"
        dx_line = " ".join([f"{int(dx):d}" if dx < 1E6 else f"{dx:.4E}"
                            for dx in model_dict['dX']])
        dy_line = " ".join([f"{int(dy):d}" if dy < 1E6 else f"{dy:.4E}"
                            for dy in model_dict['dY']])
        dz_line = " ".join([f"{int(dz):d}" if dz < 1E6 else f"{dz:.4E}"
                            for dz in model_dict['dZ']])
        lines += (f"{model_dict['cellXnum']} {model_dict['cellYnum']} "
                  f"{model_dict['cellZnum']} {model_dict['cellAirnum']} "
                  f"{model_dict['valtype']}\n{dx_line}\n{dy_line}\n{dz_line}\n\n")

    for slice_idx, slc in enumerate(slices):
        if field_name in slc.cell_data:
            values = np.asarray(slc.cell_data[field_name])
        else:
            raise KeyError(
                f"Field '{field_name}' not found in slice {slice_idx}. "
                f"Available: {list(slc.point_data.keys()) + list(slc.cell_data.keys())}")
        pts = slc.cell_centers().points
        ys = np.asarray(pts)[:, 0]
        xs = np.asarray(pts)[:, 1]
        unique_x = np.unique(xs)
        unique_y = np.unique(ys)
        nx, ny = len(unique_x), len(unique_y)
        col_idx = np.searchsorted(unique_x, xs)
        row_idx = np.searchsorted(unique_y, ys)
        grid = np.empty((ny, nx), dtype=values.dtype)
        grid[row_idx, col_idx] = values
        if flipx and flipy:
            grid = np.fliplr(np.flipud(grid))
        elif flipy:
            grid = np.fliplr(grid)
        elif flipx:
            grid = np.flipud(grid)
        if slice_idx > 0:
            lines += "\n"
        for row in reversed(range(ny)):
            if model_dict:
                if model_dict['valtype'] == "LOGE":
                    lines += "    " + "    ".join(fmt % np.log(v)
                                                  for v in grid[row, :]) + "\n"
                else:
                    lines += "    " + "    ".join(fmt % v
                                                  for v in grid[row, :]) + "\n"
    if model_dict:
        lines += " ".join([f"{int(c)}" for c in model_dict['origin']]) + "\n"
        lines += f"{int(model_dict['rot_angle'])}" + "\n"
    with open(output_path, "w") as f:
        f.write(lines)


_DEFAULT_COV_HEADER = (
    "+---------------------+\n"
    "| This file defines model covariance for a recursive autoregression scheme.   |\n"
    "| The model space may be divided into distinct areas using integer masks.     |\n"
    "| Mask 0 is reserved for air; mask 9 is reserved for ocean. Smoothing between |\n"
    "| air, ocean and the rest of the model is turned off automatically. You can   |\n"
    "| also define exceptions to override smoothing between any two model areas.   |\n"
    "| To turn off smoothing set it to zero.  This header is 16 lines long.        |\n"
    "| 1. Grid dimensions excluding air layers (Nx, Ny, NzEarth)                   |\n"
    "| 2. Smoothing in the X direction (NzEarth real values)                       |\n"
    "| 3. Smoothing in the Y direction (NzEarth real values)                       |\n"
    "| 4. Vertical smoothing (1 real value)                                        |\n"
    "| 5. Number of times the smoothing should be applied (1 integer >= 0)         |\n"
    "| 6. Number of exceptions (1 integer >= 0)                                    |\n"
    "| 7. Exceptions in the for e.g. 2 3 0. (to turn off smoothing between 3 & 4)  |\n"
    "| 8. Two integer layer indices and Nx x Ny block of masks, repeated as needed.|\n"
    "+---------------------+\n"
)


def write_slices_to_cov_file(slices, field_name, output_path, cov_dict,
                             cellAirnum=0, fmt="%d",
                             flipx=False, flipy=False) -> None:
    """Write a ModEM ``.cov`` file from pyvista slices of integer masks.

    The first ``cellAirnum`` slices are skipped (``.cov`` excludes air) and
    adjacent layers with identical masks are collapsed into one block.

    :raises ImportError: If pyvista is not installed.
    """
    _require_pyvista()
    Nx = cov_dict['cellXnum']
    Ny = cov_dict['cellYnum']
    NzEarth = cov_dict['NzEarth']

    if cov_dict.get('header_comments'):
        lines = "".join(cov_dict['header_comments'])
    else:
        lines = _DEFAULT_COV_HEADER
    lines += "\n"
    lines += f" {Nx}       {Ny}       {NzEarth}        \n\n"
    lines += " " + "  ".join(f"{v:.2f}" for v in cov_dict['smoothing_X']) + " \n"
    lines += " " + "  ".join(f"{v:.2f}" for v in cov_dict['smoothing_Y']) + " \n"
    lines += f" {cov_dict['smoothing_Z']} \n\n"
    lines += f" {int(cov_dict['num_smoothing'])} \n\n"
    lines += f" {int(cov_dict['num_exceptions'])}\n\n"
    for ex in cov_dict.get('exceptions', []):
        lines += f" {int(ex[0])} {int(ex[1])} {ex[2]}\n"
    lines += "\n"

    earth_slices = list(slices)[cellAirnum:]
    if len(earth_slices) != NzEarth:
        raise ValueError(
            f"Expected {NzEarth} earth slices (len(slices)={len(slices)}, "
            f"cellAirnum={cellAirnum}), got {len(earth_slices)}.")

    layer_blocks = []
    for slice_idx, slc in enumerate(earth_slices):
        if field_name in slc.cell_data:
            values = np.asarray(slc.cell_data[field_name])
        else:
            raise KeyError(
                f"Field '{field_name}' not found in slice {slice_idx}. "
                f"Available: {list(slc.point_data.keys()) + list(slc.cell_data.keys())}")
        pts = slc.cell_centers().points
        ys = np.asarray(pts)[:, 0]
        xs = np.asarray(pts)[:, 1]
        unique_x = np.unique(xs)
        unique_y = np.unique(ys)
        nx, ny = len(unique_x), len(unique_y)
        if nx != Nx or ny != Ny:
            raise ValueError(
                f"Slice {slice_idx} has ({nx}, {ny}) unique (X, Y) coords; "
                f"expected ({Nx}, {Ny}).")
        col_idx = np.searchsorted(unique_x, xs)
        row_idx = np.searchsorted(unique_y, ys)
        grid = np.empty((ny, nx), dtype=values.dtype)
        grid[row_idx, col_idx] = values
        if flipx and flipy:
            grid = np.fliplr(np.flipud(grid))
        elif flipy:
            grid = np.fliplr(grid)
        elif flipx:
            grid = np.flipud(grid)
        block = grid[::-1, :].T.astype(int)
        layer_blocks.append(block)

    blocks_to_write = []
    k = 0
    while k < NzEarth:
        j = k + 1
        while j < NzEarth and np.array_equal(layer_blocks[j], layer_blocks[k]):
            j += 1
        blocks_to_write.append((k + 1, j, layer_blocks[k]))
        k = j

    for block_idx, (sl, el, block) in enumerate(blocks_to_write):
        if block_idx > 0:
            lines += "\n"
        lines += f" {sl}       {el}       \n"
        for x_idx in range(block.shape[0]):
            lines += " " + "  ".join(fmt % v for v in block[x_idx, :]) + " \n"

    with open(output_path, "w") as f:
        f.write(lines)
