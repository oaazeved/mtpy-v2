""" ``makeDHexaMesh`` input builder

GIS dependencies (geopandas, rasterio, shapely) are imported lazily, only on 
the topography path.

FEMTIC / makeDHexaMesh coordinate convention: X = north,
Y = east, Z positive downward, all in km.

@author: oaazeved

"""

from __future__ import annotations

import pathlib
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from .mesh import FemticMesh, PathLike

# meshgen.inp section keywords, in write order.
_MESHGEN_KEYS = ("DIVISION_NUMBERS", "X_COORDINATES", "Y_COORDINATES",
                 "Z_COORDINATES", "SEA_DEPTH", "SEA_RESISTIVITY",
                 "THRE_SEA_DEPTH", "INITIAL_RESISTIVITY", "AIR_RESISTIVITY",
                 "CUBOIDS", "ANOMALIES", "TOPO", "END")


# Topography sampling helpers (GIS; imported lazily)

def _create_point_grid(sites_points_gdf, coast_lines_gdf=None,
                       site_buffer=3000.0, coast_lines_gdf_buffer=3000.0,
                       spacing=4000.0, crs="EPSG:32616", bounds=None):
    """Build a (optionally coast-densified) regular point grid for topo
    sampling. Lazily imports geopandas/shapely."""
    import geopandas as gpd
    import shapely.geometry as shgeom

    sites_points_gdf['buffer'] = sites_points_gdf['geometry'].buffer(site_buffer)
    if coast_lines_gdf is not None:
        if (np.any(coast_lines_gdf['geometry'].geom_type == 'Polygon')
                or np.any(coast_lines_gdf['geometry'].geom_type == 'MultiPolygon')):
            coast_lines_gdf['geometry'] = coast_lines_gdf.boundary
        coast_lines_gdf['buffer'] = coast_lines_gdf['geometry'].buffer(coast_lines_gdf_buffer)
        total_buffer = gpd.GeoDataFrame(
            geometry=pd.concat([sites_points_gdf['buffer'],
                                coast_lines_gdf['buffer']]))
    else:
        total_buffer = gpd.GeoDataFrame(geometry=sites_points_gdf['buffer'])

    total_buffer = total_buffer.dissolve()
    if bounds is None:
        minx = total_buffer.bounds.minx.min()
        miny = total_buffer.bounds.miny.min()
        maxx = total_buffer.bounds.maxx.max()
        maxy = total_buffer.bounds.maxy.max()
    else:
        minx, miny, maxx, maxy = bounds
    x_values = np.arange(minx, maxx, spacing)
    y_values = np.arange(miny, maxy, spacing)
    points_list = [shgeom.Point(x, y) for x in x_values for y in y_values]
    logger.debug("wide-spaced points created")
    points_gdf = gpd.GeoDataFrame(geometry=points_list, crs=crs)
    logger.debug("points gdf created")
    logger.debug(f"size of points gdf: {points_gdf.size}")
    return points_gdf


def _sample_elevation(points_gdf, elev_src):
    """Sample DEM elevation at each grid point, dropping nodata (< -9900)."""
    coord_list = [(x, y) for x, y in zip(points_gdf["geometry"].x,
                                         points_gdf["geometry"].y)]
    # rasterio .sample yields one array per point (length = n bands); take
    # band 0 as a scalar so 'elev' is a plain float column.
    points_gdf['elev'] = [float(h[0]) for h in elev_src.sample(coord_list)]
    points_gdf.drop(points_gdf[points_gdf['elev'] < -9900].index, inplace=True)
    logger.debug("elevation & bathymetry points sampled")
    return points_gdf


class DeformableHexMesh(FemticMesh):
    """Builder for ``makeDHexaMesh`` inputs (``meshgen.inp`` + ``obs_site.dat``).

    Parameter names mirror the historical ``femtic_d_hexamesh.write_inp`` /
    ``write_obs_site`` arguments so existing call sites translate directly.

    :param mt_df: mtpy-v2 long DataFrame.
    :type mt_df: pandas.DataFrame
    :param initial_resistivity: Starting/background land resistivity
        (ohm-m); also used as ``start_res`` for skin-depth depth sizing.
        Defaults to ``100.0``.
    :type initial_resistivity: float, optional
    :param min_x_space: Minimum X (north) cell size in the survey core
        (km), defaults to ``5.0``.
    :type min_x_space: float, optional
    :param min_y_space: Minimum Y (east) cell size in the survey core
        (km), defaults to ``5.0``.
    :type min_y_space: float, optional
    :param hgrowth: Horizontal padding-cell growth factor, defaults to
        ``1.2``.
    :type hgrowth: float, optional
    :param n_padding: Number of horizontal padding cells per side,
        defaults to ``10``.
    :type n_padding: int, optional
    :param zmethod: Vertical spacing method (``"mult"`` or ``"log"``),
        defaults to ``"mult"``.
    :type zmethod: str, optional
    :param zgrowth: Vertical growth factor for ``"mult"`` spacing,
        defaults to ``1.2``.
    :type zgrowth: float, optional
    :param n_sub_levels: Number of subsurface vertical levels (``"log"``)
        / step seed (``"mult"``), defaults to ``50``.
    :type n_sub_levels: int, optional
    :param max_depth: Absolute maximum depth (km); ``None`` derives it from
        the longest-period skin depth.
    :type max_depth: float or None, optional
    :param max_depth_skin_factor: Skin-depth multiple for the maximum depth
        when ``max_depth`` is ``None``, defaults to ``3.0``.
    :type max_depth_skin_factor: float, optional
    :param min_depth: Absolute shallowest subsurface vertex (km); ``None``
        derives it from the shortest-period skin depth.
    :type min_depth: float or None, optional
    :param min_depth_skin_factor: Skin-depth multiple for the shallowest
        vertex when ``min_depth`` is ``None``, defaults to ``0.3``.
    :type min_depth_skin_factor: float, optional
    :param z_up: Explicit list of air-layer thicknesses (km) above the
        surface; ``None`` uses ``[0.1, 0.5, 1.5, 3.5, 7.0]``.
    :type z_up: list or None, optional
    :param bounds: Optional ``[xmin, xmax, ymin, ymax]`` (km) clip on the
        horizontal vertices. Here ``x`` = **north** and ``y`` = **east**
        (matching :attr:`x_vertices`/:attr:`y_vertices`), i.e.
        ``[north_min, north_max, east_min, east_max]``.
    :type bounds: list or None, optional
    :param domain_cuboids: Nested refinement cuboids written to the
        ``CUBOIDS`` block of ``meshgen.inp``. Each row is
        ``[lenX, lenY, lenZ, edgeHorizontal]``. ``True`` writes one default
        cuboid; ``False``/``None`` writes none.
    :type domain_cuboids: list or bool or None, optional
    :param domain_cuboids_center: center of the domain cuboids (km, in
        north/east); ``None`` uses the survey bounding-box center.
    :type domain_cuboids_center: tuple or None, optional
    :param domain_cuboids_rotation: Rotation of the domain cuboids (deg),
        defaults to ``0.0``.
    :type domain_cuboids_rotation: float, optional
    :param site_cuboids: Per-site refinement cuboids written to
        ``obs_site.dat``. Each row is
        ``[width_km, max_cell_km, vertical_deform]``.
    :type site_cuboids: list, optional
    :param site_height: Site elevation written to ``obs_site.dat``,
        defaults to ``0.0``.
    :type site_height: float, optional
    :param topo_name: Optional name of a ``topo.xyz`` to reference in the
        ``TOPO`` block of ``meshgen.inp``.
    :type topo_name: str or None, optional
    :param topo_options: Topography interpolation options
        ``[n_points, search_radius, eps]``, defaults to ``[3, 15.0, 0.001]``.
    :type topo_options: list, optional
    :param sea_resistivity: Ocean-cell resistivity (ohm-m), defaults to
        ``0.3``.
    :type sea_resistivity: float, optional
    :param air_resistivity: Air-cell resistivity (ohm-m), defaults to
        ``1.0e10``.
    :type air_resistivity: float, optional
    :param anomalies: Optional anomalous-resistivity bodies for the
        ``ANOMALIES`` block.
    :type anomalies: list or None, optional
    """

    def __init__(self, mt_df: pd.DataFrame, *,
                 initial_resistivity: float = 100.0,
                 min_x_space: float = 5.0, min_y_space: float = 5.0,
                 hgrowth: float = 1.2, n_padding: int = 10,
                 zmethod: str = "mult", zgrowth: float = 1.2,
                 n_sub_levels: int = 50,
                 max_depth: Optional[float] = None,
                 max_depth_skin_factor: float = 3.0,
                 min_depth: Optional[float] = None,
                 min_depth_skin_factor: float = 0.3,
                 z_up: Optional[Sequence[float]] = None,
                 bounds: Optional[Sequence[float]] = None,
                 domain_cuboids=None, domain_cuboids_center=None,
                 domain_cuboids_rotation: float = 0.0,
                 site_cuboids=([1.0, 0.2, 0.0], [4.0, 0.5, 0.0],
                               [10.0, 2.0, 0.0]),
                 site_height: float = 0.0,
                 topo_name: Optional[str] = None,
                 topo_options=(3, 15.0, 0.001),
                 sea_resistivity: float = 0.3,
                 air_resistivity: float = 1.0e10,
                 anomalies=None):
        super().__init__(mt_df, start_res=initial_resistivity)
        self.initial_resistivity = float(initial_resistivity)
        self.min_x_space = min_x_space
        self.min_y_space = min_y_space
        self.hgrowth = hgrowth
        self.n_padding = n_padding
        self.zmethod = zmethod
        self.zgrowth = zgrowth
        self.n_sub_levels = n_sub_levels
        self.max_depth = max_depth
        self.max_depth_skin_factor = max_depth_skin_factor
        self.min_depth = min_depth
        self.min_depth_skin_factor = min_depth_skin_factor
        self.z_up = list(z_up) if z_up is not None else None
        self.bounds = list(bounds) if bounds is not None else None
        self.domain_cuboids = domain_cuboids
        self.domain_cuboids_center = domain_cuboids_center
        self.domain_cuboids_rotation = domain_cuboids_rotation
        self.site_cuboids = ([list(c) for c in site_cuboids]
                             if site_cuboids else site_cuboids)
        self.site_height = site_height
        self.topo_name = topo_name
        self.topo_options = list(topo_options)
        self.sea_resistivity = sea_resistivity
        self.air_resistivity = air_resistivity
        self.anomalies = anomalies

    # computed mesh geometry 

    def _horizontal_vertices(self, coord_col: str, min_space: float) -> np.ndarray:
        """Symmetric, padded vertex coordinates (km) along one horizontal axis.

        Vertices are uniformly spaced at ``min_space`` across the station
        extent, then grow by :attr:`hgrowth` over :attr:`n_padding` cells on
        each side, recentered on the survey midpoint.

        :param coord_col: ``"north"`` (X axis) or ``"east"`` (Y axis).
        :type coord_col: str
        :param min_space: Core cell size (km).
        :type min_space: float
        :rtype: numpy.ndarray
        """
        lo = self.mt_df[coord_col].min() / 1000.0
        hi = self.mt_df[coord_col].max() / 1000.0
        center = (hi + lo) / 2.0
        domain = hi - lo
        inner = np.arange(0.0, (domain / 2) + min_space, min_space)
        step = min_space
        outer = np.array([np.max(inner) + step])
        for i in range(self.n_padding):
            outer = np.append(outer, outer[i] + step)
            step = step * self.hgrowth
        half = np.concatenate([inner, outer])
        coords = np.concatenate([np.flip(-half[1:]), half])
        return coords + center

    @property
    def x_vertices(self) -> np.ndarray:
        """X (north) mesh-vertex coordinates (km), before any bounds clip."""
        return self._horizontal_vertices("north", self.min_x_space)

    @property
    def y_vertices(self) -> np.ndarray:
        """Y (east) mesh-vertex coordinates (km), before any bounds clip."""
        return self._horizontal_vertices("east", self.min_y_space)

    @property
    def z_vertices(self) -> np.ndarray:
        """Z (depth) mesh-vertex coordinates (km; negative is air)."""
        rho = self.initial_resistivity
        periods = self.mt_df["period"]
        max_skin_depth = 500.0 * np.sqrt(rho * np.max(periods)) / 1000.0
        min_skin_depth = 500.0 * np.sqrt(rho * np.min(periods)) / 1000.0
        max_depth = (self.max_depth_skin_factor * max_skin_depth
                     if self.max_depth is None else self.max_depth)
        min_depth = (self.min_depth_skin_factor * min_skin_depth
                     if self.min_depth is None else self.min_depth)
        if max_depth <= min_depth:
            raise ValueError(
                f"max_depth ({max_depth:.3f} km) must be greater than "
                f"min_depth ({min_depth:.3f} km).")

        if self.zmethod == "log":
            self.logger.debug("log z-spacing used")
            z_down = np.logspace(np.log10(min_depth / 2.0),
                                 np.log10(max_depth), self.n_sub_levels)
        elif self.zmethod == "mult":
            self.logger.debug(f"mult z-spacing used, x{self.zgrowth}")
            dz = min_depth
            z_down = np.array([min_depth])
            while z_down[-1] < max_depth:
                dz = dz * self.zgrowth
                z_down = np.append(z_down, z_down[-1] + dz)
        else:
            raise ValueError("zmethod must be 'mult' or 'log'.")

        z_up = [0.1, 0.5, 1.5, 3.5, 7.0] if self.z_up is None else self.z_up
        return np.concatenate([-np.flip(z_up), [0.0], z_down])

    @property
    def n_cells(self) -> tuple:
        """``(nx, ny, nz)`` cell counts implied by the (unclipped) vertices."""
        return (len(self.x_vertices) - 1, len(self.y_vertices) - 1,
                len(self.z_vertices) - 1)

    # writers 

    def write_meshgen_inp(self, path: PathLike = "meshgen.inp") -> None:
        """Write ``meshgen.inp`` for ``makeDHexaMesh``.

        :param path: Output file path, defaults to ``"meshgen.inp"``.
        :type path: str or pathlib.Path
        """
        keys = _MESHGEN_KEYS

        # `domain_cuboids=True` => one default refinement cuboid; False/None => none.
        cuboids = self.domain_cuboids
        if isinstance(cuboids, bool):
            cuboids = [[50.0, 50.0, 5.0, 0.5]] if cuboids else None

        x = self.x_vertices
        y = self.y_vertices
        if self.bounds is not None:
            x = x[(x > self.bounds[0]) & (x < self.bounds[1])]
            y = y[(y > self.bounds[2]) & (y < self.bounds[3])]
        z = self.z_vertices

        with open(path, "w") as file:
            file.write(f"{keys[0]}\n{len(x)-1} {len(y)-1} {len(z)-1}\n")
            file.write(f"{keys[1]}\n")
            for i in x:
                file.write(f"{i:.3f}\n")
            file.write(f"{keys[2]}\n")
            for i in y:
                file.write(f"{i:.3f}\n")
            file.write(f"{keys[3]}\n")
            for i in z:
                file.write(f"{i:.3f}\n")
            file.write(f"{keys[5]}\n{self.sea_resistivity:.1f}\n")
            file.write(f"{keys[7]}\n{self.initial_resistivity:.1f}\n")
            file.write(f"{keys[8]}\n{self.air_resistivity:.1e}\n")
            if cuboids is not None:
                # makeDHexaMesh expects, after the CUBOIDS keyword: a
                # center(X Y Z) line, rotation line, count, then one row per
                # cuboid. center defaults to the survey bounding-box center
                # (X=north, Y=east) so the nested cuboids sit on the array.
                if self.domain_cuboids_center is None:
                    cx = (float(self.mt_df["north"].max())
                          + float(self.mt_df["north"].min())) / 2.0 / 1000.0
                    cy = (float(self.mt_df["east"].max())
                          + float(self.mt_df["east"].min())) / 2.0 / 1000.0
                    cz = 0.0
                else:
                    cx, cy, cz = self.domain_cuboids_center
                file.write(f"{keys[9]}\n")
                file.write(f"{cx:.3f} {cy:.3f} {cz:.3f}\n")
                file.write(f"{self.domain_cuboids_rotation:.3f}\n")
                file.write(f"{len(cuboids)}\n")
                for c in cuboids:
                    file.write(" ".join(f"{v:.3f}" for v in c) + " \n")
            if self.anomalies is not None:
                file.write(f"{keys[10]}\n")
                file.write(f"{len(self.anomalies)}\n")
                for a in self.anomalies:
                    for i in a:
                        file.write(f"{i:7} ")
                    file.write("\n")
            if self.topo_name is not None:
                file.write(f"{keys[11]}\n")
                file.write(f"{self.topo_name}\n")
                for i in self.topo_options:
                    file.write(f"{i}\n")
            file.write(f"{keys[12]}\n\n")
        self.logger.info("meshgen.inp written")

    def write_obs_site(self, out_dir: PathLike) -> None:
        """Write ``obs_site.dat`` (site locations + per-site refinement cuboids).

        :param out_dir: Destination directory.
        :type out_dir: str or pathlib.Path
        """
        cuboids = self.site_cuboids
        with open(pathlib.Path(out_dir).joinpath("obs_site.dat"), "w") as file:
            if cuboids is None or len(cuboids) == 0 or len(cuboids[0]) == 0:
                file.write("0 ")
            else:
                file.write(f"{self.n_stations}\n")
                for s in self.stations:
                    station = self.mt_df[self.mt_df["station"] == s].reset_index()
                    # FEMTIC/makeDHexaMesh convention: X = north, then Y = east.
                    file.write(
                        f"{station['north'].iloc[0]/1000.0:.4f}  "
                        f"{station['east'].iloc[0]/1000.0:.4f}  "
                        f"{self.site_height}   {len(cuboids)} ")
                    for c in cuboids:
                        file.write(f" {c[0]}  {c[1]} {c[2]}")
                    file.write("\n")
        self.logger.info("obs_site.dat written")

    def write_topo_xyz(self, dem_path: PathLike,
                       out_path: PathLike = "topo.xyz",
                       crs: str = "EPSG:32616", spacing: float = 1000.0,
                       coast_path: Optional[PathLike] = None):
        """Write a ``topo.xyz`` sampled from a DEM (needs geopandas/rasterio).

        :param dem_path: Path to the DEM raster.
        :type dem_path: str or pathlib.Path
        :param out_path: Output ``topo.xyz`` path, defaults to ``"topo.xyz"``.
        :type out_path: str or pathlib.Path
        :param crs: CRS of the MT data, defaults to ``"EPSG:32616"``.
        :type crs: str, optional
        :param spacing: Background sampling spacing (m), defaults to ``1000.0``.
        :type spacing: float, optional
        :param coast_path: Optional coast file for denser coastal sampling.
        :type coast_path: str or pathlib.Path or None, optional
        :return: The sampled topo bounding box (km).
        """
        import geopandas as gpd
        import rasterio as rt
        import shapely.geometry as shgeom

        site_pts_gdf = gpd.GeoDataFrame(
            geometry=[shgeom.Point(x, y) for x, y in
                      zip(self.mt_df["east"] / 1000.0,
                          self.mt_df["north"] / 1000.0)])
        if coast_path is not None:
            coast_poly_gdf = gpd.read_file(coast_path)
            pts_gdf = _create_point_grid(site_pts_gdf, coast_poly_gdf,
                                         site_buffer=1.0,
                                         coast_lines_gdf_buffer=1.0,
                                         spacing=spacing, bounds=None, crs=crs)
        else:
            pts_gdf = _create_point_grid(site_pts_gdf, site_buffer=1.0,
                                         coast_lines_gdf_buffer=1.0,
                                         spacing=spacing, bounds=None, crs=crs)

        with rt.open(dem_path) as src:
            topo_pts = _sample_elevation(pts_gdf, src)
        with open(out_path, "w") as file:
            for _, row in topo_pts.iterrows():
                # X = north (geometry.y), Y = east (geometry.x); Z down.
                file.write(f"{row['geometry'].y/1000.0:.8f} "
                           f"{row['geometry'].x/1000.0:.8f} "
                           f"{-float(row['elev'])/1000.0:.8f} \n")
        self.logger.info(f"{out_path} written")
        return topo_pts["geometry"].total_bounds / 1000.0

    def write_inputs(self, out_dir: PathLike,
                     dem_path: Optional[PathLike] = None) -> dict:
        """Write ``meshgen.inp`` and ``obs_site.dat`` (and ``topo.xyz`` if a
        DEM is given) into ``out_dir``.

        :param out_dir: Destination directory (created if missing).
        :type out_dir: str or pathlib.Path
        :param dem_path: Optional DEM raster to also produce ``topo.xyz``.
        :type dem_path: str or pathlib.Path or None, optional
        :return: ``{filename: path}`` for every file written.
        :rtype: dict
        """
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        meshgen = out_dir / "meshgen.inp"
        self.write_meshgen_inp(meshgen)
        written["meshgen.inp"] = meshgen
        self.write_obs_site(out_dir)
        written["obs_site.dat"] = out_dir / "obs_site.dat"
        if dem_path is not None:
            topo = out_dir / (self.topo_name or "topo.xyz")
            self.write_topo_xyz(dem_path, out_path=topo)
            written[topo.name] = topo
        return written

    @staticmethod
    def read_meshgen_inp(path: PathLike) -> dict:
        """Parse a ``meshgen.inp`` into a keyword -> values dict.

        Untested reader, ported as-is from the original module.

        :param path: Path to ``meshgen.inp``.
        :type path: str or pathlib.Path
        :return: Parsed mapping.
        :rtype: dict
        """
        with open(path, "r") as file:
            lines = [line.rstrip() for line in file]
        inp_dict = {}
        current_key = None
        for line in lines:
            try:
                float(line)
                inp_dict[current_key].append(float(line))
            except (ValueError, TypeError):
                if len(line.split()) < 2:
                    if len(line.split(".")) > 1:
                        inp_dict[current_key].append(line)
                    else:
                        inp_dict[line] = []
                        current_key = line
                else:
                    inp_dict[current_key].append(
                        [float(part) for part in line.split()])
        return inp_dict
