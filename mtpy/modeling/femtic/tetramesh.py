"""Object-oriented ``makeTetraMesh`` pipeline input builder.

You may want to exclude this module for now. This is easily the least-tested 
portion of the femtic submodule. Every mesh  I've tried to assemble has 
crashed by step 3 of makeTetraMesh, at the latest; this is likely a problem 
with `coastline.dat` and `topo.xyz` processing causing duplicates or 
crossings/overlaps.

:class:`TetraMesh` holds an mtpy-v2 station DataFrame plus the tetra-mesh
configuration and writes the eight FEMTIC tetra starting files consumed by the
``makeTetraMesh`` / ``TetGen`` / ``makeMtr`` / ``TetGen2Femtic`` pipeline.

Coordinate conventions

* **Model frame** (centering, bounds, station coordinates, every FEMTIC
  output file, and the public :class:`TetraMesh` API): ``x = north``,
  ``y = east``, ``z`` positive **down**. This matches FEMTIC's file format
  and :mod:`hexmesh`. So :attr:`TetraMesh.model_center` is
  ``(north_m, east_m)`` and :attr:`TetraMesh.bounds_km` is
  ``(north_min, north_max, east_min, east_max, z_air, z_earth)``.
* **GIS pipeline** (raster/xyz topo loaders, coast polygonisation, the
  variable-density sampling grid, ring orientation): these operate in
  shapely/rasterio space, where ``x = easting`` is necessary. To keep 
  that unambiguous, those DataFrames now use explicit ``east_m`` / ``north_m`` 
  columns (= projected easting / northing), and shapely points are 
  ``Point(east_m, north_m)``. The conversion to the model frame (north 
  first) happens explicitly inside the file writers.

Z conventions by file (should be same as FEMTIC):

* altitude file (``ALTITUDE``): positive **up** (a +500 m hilltop -> +0.500);
* bathymetry file (``SEA_DEPTH``): positive **down**;
* station Z in ``obs_site.dat`` / ``resistivity_attr.dat``: positive into
  the earth, so the +500 m hilltop is ``-0.500``.

The GIS dependencies (geopandas, rasterio, shapely, scipy, pyproj) are
imported lazily, only on the paths that need them.

@author: oaazeved

"""

from __future__ import annotations

import warnings
import pathlib

import numpy as np
import pandas as pd
from loguru import logger

from .mesh import FemticMesh, PathLike

# Skin-depth constant: depth [m] ~= 500 * sqrt(rho[ohm-m] * T[s])
_SKIN_DEPTH_CONST = 500.0


def _require_geopandas():
    import geopandas as gpd
    import shapely.geometry as shgeom
    from shapely.ops import unary_union
    return gpd, shgeom, unary_union

def _require_rasterio():
    import rasterio as rt
    from rasterio import features as rt_features
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    return rt, rt_features, reproject, Resampling, calculate_default_transform

def _has_geopandas() -> bool:
    try:
        import geopandas
        import shapely
        return True
    except ImportError:
        return False



# GIS pipeline; definitely needs some work.
# Coordinates here are (east_m = projected easting = shapely X,
#                      north_m = projected northing = shapely Y).
# This is shapely/rasterio convention, separate from the model 
# frame (x=north) used everywhere else.

def _detect_topo_kind(path):
    """Guess ``'xyz'`` vs ``'raster'`` from a topo file's extension."""
    ext = pathlib.Path(path).suffix.lower()
    if ext in ('.xyz', '.txt', '.csv', '.dat'):
        return 'xyz'
    return 'raster'


def _load_topo_xyz(path, delimiter=None, header=False, columns=(0, 1, 2),
                   z_units='m', src_crs=None, dst_crs=None):
    """Read an easting/northing/elevation text file.

    :param columns: Three column indices selecting ``(east, north, z)`` from
        the file, defaults to ``(0, 1, 2)``.
    :returns: DataFrame with columns ``east_m``, ``north_m``, ``z_m``
        (elevation positive up, meters).
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(path, sep=delimiter if delimiter is not None else r'\s+',
                     header=0 if header else None, comment='#', engine='python')
    ce, cn, cz = columns
    out = pd.DataFrame({
        'east_m':  df.iloc[:, ce].astype(float).to_numpy(),
        'north_m': df.iloc[:, cn].astype(float).to_numpy(),
        'z_m':     df.iloc[:, cz].astype(float).to_numpy(),
    })
    if z_units == 'km':
        out['z_m'] *= 1000.0
    if dst_crs is not None:
        if src_crs is None:
            emin, emax = float(out['east_m'].min()), float(out['east_m'].max())
            nmin, nmax = float(out['north_m'].min()), float(out['north_m'].max())
            if -180.0 <= emin and emax <= 180.0 and -90.0 <= nmin and nmax <= 90.0:
                src_crs = 'EPSG:4326'
        if src_crs is not None and str(src_crs) != str(dst_crs):
            from pyproj import Transformer
            tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            ne, nn = tr.transform(out['east_m'].to_numpy(),
                                  out['north_m'].to_numpy())
            out['east_m'] = np.asarray(ne, dtype=float)
            out['north_m'] = np.asarray(nn, dtype=float)
    return out


def _decimate_xyz_to_spacing(df, target_spacing_m):
    """Bucket-decimate scattered topo to ~one point per cell."""
    if target_spacing_m is None or target_spacing_m <= 0:
        return df
    be = np.round(df['east_m'].to_numpy() / target_spacing_m).astype(np.int64)
    bn = np.round(df['north_m'].to_numpy() / target_spacing_m).astype(np.int64)
    keys = np.stack([be, bn], axis=1)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def _load_topo_raster(path, dst_crs=None, target_spacing_m=None,
                      bounds_m=None, nodata=None):
    """Read a raster topo file via rasterio.

    :param bounds_m: Optional ``(east_min, north_min, east_max, north_max)``
        clip box in the destination CRS.
    :returns: DataFrame with columns ``east_m``, ``north_m``, ``z_m``.
    """
    rt, _, reproject, Resampling, calculate_default_transform = _require_rasterio()
    with rt.open(path) as src:
        src_crs = src.crs
        if dst_crs is not None and src_crs is not None and \
                str(src_crs) != str(dst_crs):
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height, *src.bounds)
            data = np.empty((height, width), dtype=np.float32)
            reproject(source=rt.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src_crs,
                      dst_transform=transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear)
            src_nodata = src.nodata
        else:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            src_nodata = src.nodata

        px = abs(transform.a)
        py = abs(transform.e)
        if target_spacing_m is not None and target_spacing_m > 0:
            stride_x = max(1, int(round(target_spacing_m / px)))
            stride_y = max(1, int(round(target_spacing_m / py)))
        else:
            stride_x = stride_y = 1

        rows = np.arange(0, data.shape[0], stride_y)
        cols = np.arange(0, data.shape[1], stride_x)
        col_grid, row_grid = np.meshgrid(cols, rows)
        # rasterio transform.xy returns (easting, northing)
        es, ns = rt.transform.xy(transform, row_grid.ravel(),
                                 col_grid.ravel(), offset='center')
        es = np.asarray(es, dtype=float)
        ns = np.asarray(ns, dtype=float)
        zs = data[row_grid, col_grid].ravel().astype(float)

        sentinel = nodata if nodata is not None else src_nodata
        keep = np.isfinite(zs)
        if sentinel is not None:
            keep &= (zs != sentinel)
        keep &= (zs > -9000.0)
        df = pd.DataFrame({'east_m': es[keep], 'north_m': ns[keep],
                           'z_m': zs[keep]})

    if bounds_m is not None:
        emin, nmin, emax, nmax = bounds_m
        df = df[(df['east_m'] >= emin) & (df['east_m'] <= emax) &
                (df['north_m'] >= nmin) & (df['north_m'] <= nmax)
                ].reset_index(drop=True)
    return df


def _load_topo(topo_path, kind='auto', dst_crs=None, target_spacing_m=None,
               bounds_m=None, **kwargs):
    """Load topography into a tidy ``east_m, north_m, z_m`` DataFrame.

    :param bounds_m: Optional ``(east_min, north_min, east_max, north_max)``
        clip box (destination CRS).
    """
    if kind == 'auto':
        kind = _detect_topo_kind(topo_path)
    if kind == 'xyz':
        df = _load_topo_xyz(topo_path, dst_crs=dst_crs, **kwargs)
        if bounds_m is not None:
            emin, nmin, emax, nmax = bounds_m
            df = df[(df['east_m'] >= emin) & (df['east_m'] <= emax) &
                    (df['north_m'] >= nmin) & (df['north_m'] <= nmax)
                    ].reset_index(drop=True)
        if target_spacing_m is not None and target_spacing_m > 0:
            df = _decimate_xyz_to_spacing(df, target_spacing_m)
        return df
    if kind == 'raster':
        return _load_topo_raster(topo_path, dst_crs=dst_crs,
                                 target_spacing_m=target_spacing_m,
                                 bounds_m=bounds_m)
    raise ValueError(f"Unknown topography kind: {kind!r}")


def _create_topo_point_grid(mt_df_centered, coast_gdf=None,
                            coarse_spacing_m=4000.0, dense_spacing_m=400.0,
                            site_buffer_m=3000.0, coast_buffer_m=3000.0,
                            bounds_m=None, crs=None):
    """Variable-density sampling grid (centered model coords, meters).

    :param bounds_m: Optional ``(east_min, north_min, east_max, north_max)``.
    :returns: GeoDataFrame/DataFrame with ``east_m``, ``north_m`` columns.
    """
    have_gpd = _has_geopandas()
    if have_gpd:
        gpd, shgeom, unary_union = _require_geopandas()

    east = mt_df_centered['east'].to_numpy(dtype=float)
    north = mt_df_centered['north'].to_numpy(dtype=float)

    if bounds_m is None:
        pad = max(site_buffer_m, coast_buffer_m) + coarse_spacing_m
        emin, emax = east.min() - pad, east.max() + pad
        nmin, nmax = north.min() - pad, north.max() + pad
    else:
        emin, nmin, emax, nmax = bounds_m

    es_c = np.arange(emin, emax + coarse_spacing_m, coarse_spacing_m)
    ns_c = np.arange(nmin, nmax + coarse_spacing_m, coarse_spacing_m)
    EC, NC = np.meshgrid(es_c, ns_c)
    coarse = np.column_stack([EC.ravel(), NC.ravel()])

    dense_pts = []
    if have_gpd:
        station_pts = [shgeom.Point(e, n) for e, n in zip(east, north)]
        buffers = [p.buffer(site_buffer_m) for p in station_pts]
        if coast_gdf is not None and len(coast_gdf) > 0:
            coast_lines = coast_gdf.geometry.apply(
                lambda g: g.boundary if g.geom_type in
                ('Polygon', 'MultiPolygon') else g)
            buffers.extend([g.buffer(coast_buffer_m) for g in coast_lines])
        dense_region = unary_union(buffers)
        demin, dnmin, demax, dnmax = dense_region.bounds
        es_d = np.arange(demin, demax + dense_spacing_m, dense_spacing_m)
        ns_d = np.arange(dnmin, dnmax + dense_spacing_m, dense_spacing_m)
        ED, ND = np.meshgrid(es_d, ns_d)
        candidates = np.column_stack([ED.ravel(), ND.ravel()])
        from shapely.prepared import prep
        prepared = prep(dense_region)
        mask = np.array([prepared.contains(shgeom.Point(e, n))
                         for e, n in candidates])
        dense_pts = candidates[mask]
    else:
        if coast_gdf is not None:
            raise ImportError("geopandas/shapely are required to densify "
                              "around coastlines.")
        for se, sn in zip(east, north):
            es_d = np.arange(se - site_buffer_m, se + site_buffer_m,
                             dense_spacing_m)
            ns_d = np.arange(sn - site_buffer_m, sn + site_buffer_m,
                             dense_spacing_m)
            ED, ND = np.meshgrid(es_d, ns_d)
            dense_pts.append(np.column_stack([ED.ravel(), ND.ravel()]))
        dense_pts = np.vstack(dense_pts) if dense_pts else np.empty((0, 2))

    all_pts = np.vstack([coarse, dense_pts]) if len(dense_pts) else coarse

    bucket = max(dense_spacing_m * 0.1, 1.0)
    keys = (np.round(all_pts[:, 0] / bucket).astype(np.int64),
            np.round(all_pts[:, 1] / bucket).astype(np.int64))
    _, unique_idx = np.unique(np.stack(keys, axis=1), axis=0, return_index=True)
    all_pts = all_pts[np.sort(unique_idx)]

    df = pd.DataFrame({'east_m': all_pts[:, 0], 'north_m': all_pts[:, 1]})
    if have_gpd:
        return gpd.GeoDataFrame(
            df, geometry=[shgeom.Point(e, n) for e, n in all_pts], crs=crs)
    return df


def _distance_scale_factor(east, north, scale_m, power, center):
    """Area-threshold multiplier ``1 + (d/scale)**power`` at ``(east, north)``.

    ``center`` is ``(east_c, north_c)`` in CRS units."""
    if not scale_m or scale_m <= 0:
        return (1.0 if np.isscalar(east)
                else np.ones_like(np.asarray(east, dtype=float)))
    ec, nc = center
    ea = np.asarray(east, dtype=float)
    na = np.asarray(north, dtype=float)
    d = np.hypot(ea - ec, na - nc)
    return 1.0 + (d / scale_m) ** power


def _filter_polygon_holes(poly, min_lake_area_m2, min_vertices,
                          distance_scale_m, distance_power, distance_center):
    _, shgeom, _ = _require_geopandas()
    kept_rings = []
    for ring in poly.interiors:
        try:
            ring_poly = shgeom.Polygon(ring.coords)
        except Exception:
            continue
        if not ring_poly.is_valid or ring_poly.is_empty:
            continue
        if min_vertices is not None and min_vertices > 0 \
                and len(ring.coords) < min_vertices:
            continue
        if min_lake_area_m2 is not None and min_lake_area_m2 > 0:
            c = ring_poly.centroid
            scale = _distance_scale_factor(
                c.x, c.y, distance_scale_m, distance_power, distance_center)
            if ring_poly.area < min_lake_area_m2 * scale:
                continue
        kept_rings.append(list(ring.coords))
    return shgeom.Polygon(poly.exterior.coords, kept_rings)


def _filter_holes(geom, min_lake_area_m2, min_vertices,
                  distance_scale_m, distance_power, distance_center):
    _, shgeom, _ = _require_geopandas()
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == 'Polygon':
        return _filter_polygon_holes(geom, min_lake_area_m2, min_vertices,
                                     distance_scale_m, distance_power,
                                     distance_center)
    if geom.geom_type == 'MultiPolygon':
        new_polys = [_filter_polygon_holes(p, min_lake_area_m2, min_vertices,
                                           distance_scale_m, distance_power,
                                           distance_center)
                     for p in geom.geoms]
        new_polys = [p for p in new_polys if not p.is_empty]
        if not new_polys:
            return shgeom.Polygon()
        return shgeom.MultiPolygon(new_polys)
    return geom


def _derive_coast_gdf_from_df(topo_df, sea_level=0.0, dst_crs=None,
                              grid_spacing_m=None, max_nodata_distance_m=None,
                              simplify_tol_m=None, min_island_area_m2=1.0e6,
                              min_lake_area_m2=None, min_vertices=None,
                              distance_scale_m=None, distance_power=2.0,
                              distance_center=(0.0, 0.0)):
    """Polygonise a topo DataFrame (``east_m, north_m, z_m``) into land polygons.

    ``distance_center`` is ``(east_c, north_c)`` in CRS units. Returns a
    GeoDataFrame of land polygons in shapely (easting, northing) space.
    """
    rt, rt_features, _, _, _ = _require_rasterio()
    gpd, shgeom, _ = _require_geopandas()
    if len(topo_df) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=dst_crs)

    if grid_spacing_m is not None and grid_spacing_m > 0:
        from scipy.spatial import cKDTree
        e = topo_df['east_m'].to_numpy()
        n = topo_df['north_m'].to_numpy()
        z = topo_df['z_m'].to_numpy()
        emin, emax = float(e.min()), float(e.max())
        nmin, nmax = float(n.min()), float(n.max())
        ne_ = max(2, int(np.ceil((emax - emin) / grid_spacing_m)) + 1)
        nn_ = max(2, int(np.ceil((nmax - nmin) / grid_spacing_m)) + 1)
        es = np.linspace(emin, emax, ne_)
        ns = np.linspace(nmin, nmax, nn_)
        EE, NN = np.meshgrid(es, ns)
        tree = cKDTree(np.column_stack([e, n]))
        d, idx = tree.query(np.column_stack([EE.ravel(), NN.ravel()]), k=1)
        grid = z[idx].reshape(nn_, ne_).astype(np.float32)
        effective_nodata = max_nodata_distance_m
        if effective_nodata is None and len(e) > 10:
            rng = np.random.default_rng(0)
            n_sample = min(2000, len(e))
            sample_idx = rng.choice(len(e), size=n_sample, replace=False)
            sample_pts = np.column_stack([e[sample_idx], n[sample_idx]])
            nn = tree.query(sample_pts, k=2)[0][:, 1]
            effective_nodata = 3.0 * float(np.median(nn))
        if effective_nodata is not None and effective_nodata > 0:
            too_far = d.reshape(nn_, ne_) > effective_nodata
            grid[too_far] = np.nan
        de = float(es[1] - es[0])
        dn = float(ns[1] - ns[0])
        grid = grid[::-1, :]
        transform = rt.transform.from_origin(emin - de / 2.0,
                                             nmax + dn / 2.0, de, dn)
    else:
        es = np.sort(topo_df['east_m'].unique())
        ns = np.sort(topo_df['north_m'].unique())
        if len(es) < 2 or len(ns) < 2:
            return gpd.GeoDataFrame(geometry=[], crs=dst_crs)
        de = float(np.median(np.diff(es)))
        dn = float(np.median(np.diff(ns)))
        emin = es[0] - de / 2.0
        nmax = ns[-1] + dn / 2.0
        transform = rt.transform.from_origin(emin, nmax, de, dn)
        ne_ = len(es)
        nn_ = len(ns)
        grid = np.full((nn_, ne_), np.nan, dtype=np.float32)
        ie = np.round((topo_df['east_m'].to_numpy() - es[0]) / de).astype(int)
        iN = np.round((ns[-1] - topo_df['north_m'].to_numpy()) / dn).astype(int)
        inside = (ie >= 0) & (ie < ne_) & (iN >= 0) & (iN < nn_)
        grid[iN[inside], ie[inside]] = topo_df['z_m'].to_numpy()[inside]

    mask = (grid >= sea_level).astype(np.uint8)
    mask[np.isnan(grid)] = 0
    polys = [shgeom.shape(g) for g, v
             in rt_features.shapes(mask, mask=mask.astype(bool),
                                   transform=transform) if v == 1]
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=dst_crs)
    gdf = gpd.GeoDataFrame(geometry=polys, crs=dst_crs)
    if len(gdf) == 0:
        return gdf

    if min_vertices is not None and min_vertices > 0:
        def _exterior_vcount(g):
            if g.geom_type == 'Polygon':
                return len(g.exterior.coords)
            if g.geom_type == 'MultiPolygon':
                return max((len(p.exterior.coords) for p in g.geoms), default=0)
            return 0
        keep = gdf.geometry.apply(_exterior_vcount).to_numpy() >= min_vertices
        gdf = gdf[keep].reset_index(drop=True)

    if min_island_area_m2 is not None and min_island_area_m2 > 0 and len(gdf):
        cents = gdf.geometry.centroid
        scaling = _distance_scale_factor(
            cents.x.to_numpy(), cents.y.to_numpy(),
            distance_scale_m, distance_power, distance_center)
        areas = gdf.geometry.area.to_numpy()
        keep = areas >= (min_island_area_m2 * scaling)
        gdf = gdf[keep].reset_index(drop=True)

    if (min_lake_area_m2 is not None or
            (min_vertices is not None and min_vertices > 0)) and len(gdf):
        gdf['geometry'] = gdf.geometry.apply(
            lambda g: _filter_holes(
                g, min_lake_area_m2=min_lake_area_m2, min_vertices=min_vertices,
                distance_scale_m=distance_scale_m, distance_power=distance_power,
                distance_center=distance_center))
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    if simplify_tol_m is not None and simplify_tol_m > 0:
        gdf['geometry'] = gdf.geometry.simplify(simplify_tol_m,
                                                preserve_topology=True)
    return gdf


def _signed_area(coords):
    """Signed area of a closed ring (shoelace); + for CCW, - for CW."""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _orient_coast_polys(gdf, land_on_right=True):
    """Return ``[(coords, is_closed), ...]`` oriented per FEMTIC.

    Operates in shapely (easting, northing) space; coords are ``(N, 2)``
    arrays of ``(east_m, north_m)``."""
    rings = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'Polygon':
            polys = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polys = list(geom.geoms)
        elif geom.geom_type in ('LineString', 'MultiLineString'):
            lines = ([geom] if geom.geom_type == 'LineString'
                     else list(geom.geoms))
            for ln in lines:
                coords = np.asarray(ln.coords)
                closed = bool(np.allclose(coords[0], coords[-1]))
                rings.append((coords, closed))
            continue
        else:
            continue
        for poly in polys:
            ext = np.asarray(poly.exterior.coords)
            if land_on_right and _signed_area(ext) > 0:
                ext = ext[::-1]
            elif (not land_on_right) and _signed_area(ext) < 0:
                ext = ext[::-1]
            rings.append((ext, True))
            for hole in poly.interiors:
                inner = np.asarray(hole.coords)
                if land_on_right and _signed_area(inner) < 0:
                    inner = inner[::-1]
                elif (not land_on_right) and _signed_area(inner) > 0:
                    inner = inner[::-1]
                rings.append((inner, True))
    return rings


def _separate_ring_pinch_points(rings, min_gap_m=500.0, coincidence_tol_m=1.0,
                                max_passes=5):
    """Pull apart vertices shared by two different rings (non-manifold pinches)."""
    if min_gap_m is None or min_gap_m <= 0 or len(rings) < 2:
        return rings, 0
    rings = [(np.array(c, dtype=float), bool(closed)) for c, closed in rings]
    quant = max(float(coincidence_tol_m), 1e-9)

    def _wrapped(coords, closed):
        return (closed and len(coords) >= 2 and np.allclose(coords[0], coords[-1]))

    def _centroid(coords, closed):
        pts = coords[:-1] if _wrapped(coords, closed) else coords
        return pts.mean(axis=0)

    total_fixed = 0
    for _ in range(max_passes):
        loc2refs = {}
        for ri, (coords, closed) in enumerate(rings):
            last = len(coords) - 1 if _wrapped(coords, closed) else len(coords)
            for vi in range(last):
                key = (round(coords[vi, 0] / quant), round(coords[vi, 1] / quant))
                loc2refs.setdefault(key, []).append((ri, vi))
        moved = 0
        for refs in loc2refs.values():
            rings_here = {ri for ri, _ in refs}
            if len(rings_here) < 2:
                continue
            keep = max(rings_here, key=lambda r: len(rings[r][0]))
            for ri, vi in refs:
                if ri == keep:
                    continue
                coords, closed = rings[ri]
                v_old = coords[vi].copy()
                direction = _centroid(coords, closed) - v_old
                norm = float(np.hypot(direction[0], direction[1]))
                if norm < 1e-9:
                    direction = np.array([1.0, 0.0])
                    norm = 1.0
                coords[vi] = v_old + (direction / norm) * float(min_gap_m)
                if vi == 0 and _wrapped(np.vstack([v_old, coords[-1]]), closed):
                    coords[-1] = coords[0]
                moved += 1
        total_fixed += moved
        if moved == 0:
            break
    if total_fixed:
        warnings.warn(
            "coast_line: separated {} pinch vertex(es) shared between distinct "
            "rings by {:g} m to remove non-manifold contacts that crash "
            "makeTetraMesh step 2.".format(total_fixed, float(min_gap_m)),
            stacklevel=2)
    return rings, total_fixed



# model "frame"  (x = north, y = east, z = down)

def _model_center(mt_df, mode='bbox'):
    """Model origin ``(north0_m, east0_m)`` in source meters (x=north)."""
    if mode == 'mean':
        north0 = float(mt_df['north'].mean())
        east0 = float(mt_df['east'].mean())
    else:
        north0 = float((mt_df['north'].max() + mt_df['north'].min()) / 2.0)
        east0 = float((mt_df['east'].max() + mt_df['east'].min()) / 2.0)
    return north0, east0


def _recenter(mt_df, center=None, mode='bbox'):
    """Shift so the model center is ``(0, 0)``; ``center`` is ``(north0, east0)``."""
    if center is None:
        center = _model_center(mt_df, mode=mode)
    north0, east0 = center
    df = mt_df.copy()
    df['north'] = df['north'].astype(float) - north0
    df['east'] = df['east'].astype(float) - east0
    return df, (north0, east0)


def _auto_bounds_km(mt_df_centered, padding_km=50.0, max_period=None,
                    start_res=100.0, depth_factor=3.0, air_factor=1.0,
                    min_z_km=None, max_z_km=None):
    """Analysis-domain bounds in centered km, north-first (FEMTIC X, Y, Z order).

    :returns: ``(north_min, north_max, east_min, east_max, z_air, z_earth)``.
    """
    north_km = mt_df_centered['north'].to_numpy() / 1000.0
    east_km = mt_df_centered['east'].to_numpy() / 1000.0
    north_min = float(north_km.min()) - padding_km
    north_max = float(north_km.max()) + padding_km
    east_min = float(east_km.min()) - padding_km
    east_max = float(east_km.max()) + padding_km
    if max_z_km is None or min_z_km is None:
        if max_period is None:
            max_period = float(mt_df_centered['period'].max())
        max_skin_km = _SKIN_DEPTH_CONST * np.sqrt(start_res * max_period) / 1000.0
        if max_z_km is None:
            max_z_km = depth_factor * max_skin_km
        if min_z_km is None:
            min_z_km = -air_factor * max_z_km
    return (north_min, north_max, east_min, east_max,
            float(min_z_km), float(max_z_km))


def _write_analysis_domain(path, bounds_km):
    """Write ``analysis_domain.dat`` (bounds_km is north-first)."""
    north_min, north_max, east_min, east_max, z_min, z_max = bounds_km
    with open(path, 'w') as f:
        f.write(f"{north_min:.4f} {north_max:.4f}\n")   # i repeat: FEMTIC X = north
        f.write(f"{east_min:.4f} {east_max:.4f}\n")     # yet again: FEMTIC Y = east
        f.write(f"{z_min:.4f} {z_max:.4f}\n")


def _write_control(path, ellipsoids, center_km=(0.0, 0.0, 0.0),
                   rotation_deg=0.0, num_threads=1, surf_mesh=True,
                   altitude_file=None, alt_range_km=(0.0, 1.0e20),
                   sea_depth_file=None, depth_range_km=(0.01, 1.0e20),
                   interpolate=None):
    """Write ``control.dat``. ``center_km`` is ``(north, east, z)``."""
    nx, ey, z = center_km    # i can't emphasize enough: x=north, y=east
    with open(path, 'w') as f:
        f.write("CENTER\n")
        f.write(f"{nx:.4f} {ey:.4f} {z:.4f}\n")
        f.write("ROTATION\n")
        f.write(f"{rotation_deg:.4f}\n")
        f.write("NUM_THREADS\n")
        f.write(f"{int(num_threads)}\n")
        if surf_mesh:
            f.write("SURF_MESH\n")
        f.write("ELLIPSOIDS\n")
        f.write(f"{len(ellipsoids)}\n")
        for e in ellipsoids:
            a, ln, fh, fvp, fvm = e
            f.write(f"{a:8.3f} {ln:8.3f} {fh:6.3f} {fvp:6.3f} {fvm:6.3f}\n")
        if interpolate is not None:
            r, n, eps = interpolate
            f.write("INTERPOLATE\n")
            f.write(f"{r:.4f}\n{int(n):d}\n{eps:.4e}\n")
        if altitude_file is not None:
            amin, amax = alt_range_km
            f.write("ALTITUDE\n")
            f.write(f"{altitude_file}\n")
            f.write(f"{amin:.4f}\n{amax:.4e}\n")
        if sea_depth_file is not None:
            dmin, dmax = depth_range_km
            f.write("SEA_DEPTH\n")
            f.write(f"{sea_depth_file}\n")
            f.write(f"{dmin:.4f}\n{dmax:.4e}\n")
        f.write("END\n")


def _write_observing_site(mt_df_centered, path, spheres=None):
    """Write ``observing_site.dat`` (per-station X=north, Y=east in km)."""
    if spheres is None:
        spheres = [[0.1, 0.02], [0.3, 0.05], [1.0, 0.10],
                   [3.0, 0.30], [5.0, 0.50]]
    stations = mt_df_centered['station'].drop_duplicates().tolist()
    with open(path, 'w') as f:
        f.write(f"{len(stations)}\n")
        for s in stations:
            row = mt_df_centered.loc[mt_df_centered['station'] == s].iloc[0]
            north_km = float(row['north']) / 1000.0
            east_km = float(row['east']) / 1000.0
            f.write(f"{north_km:>14.9f} {east_km:>14.9f}  {len(spheres)}")
            for r, ln in spheres:
                f.write(f"  {r} {ln}")
            f.write("\n")


def _write_coast_line(coast_gdf, path, model_center_m=(0.0, 0.0),
                      bounds_km=None, simplify_tol_m=None, land_on_right=True,
                      min_ring_gap_m=500.0, pinch_tol_m=1.0):
    """Write ``coast_line.dat``.

    ``model_center_m`` is ``(north0_m, east0_m)``; ``bounds_km`` is
    ``(north_min, north_max, east_min, east_max)`` in centered km. Coast
    geometry is in shapely (easting, northing) space.
    """
    gpd, shgeom, _ = _require_geopandas()
    north0, east0 = model_center_m

    gdf = coast_gdf.copy()
    if simplify_tol_m is not None and simplify_tol_m > 0:
        gdf['geometry'] = gdf.geometry.simplify(simplify_tol_m,
                                                preserve_topology=True)
    if bounds_km is not None:
        north_min, north_max, east_min, east_max = bounds_km
        # clip box in source (easting, northing): box(minx, miny, maxx, maxy)
        box = shgeom.box(east_min * 1000.0 + east0, north_min * 1000.0 + north0,
                         east_max * 1000.0 + east0, north_max * 1000.0 + north0)
        gdf['geometry'] = gdf.geometry.intersection(box)
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    rings = _orient_coast_polys(gdf, land_on_right=land_on_right)
    rings, _ = _separate_ring_pinch_points(
        rings, min_gap_m=min_ring_gap_m, coincidence_tol_m=pinch_tol_m)

    with open(path, 'w') as f:
        f.write(f"{len(rings)}\n")
        for coords, is_closed in rings:
            if is_closed and len(coords) >= 2 and np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            # _orient_coast_polys established land-on-right in shapely
            # (easting-X, northing-Y) space. Writing as (north, east) is a
            # reflection of that frame, which I'm pretty sure reverses the 
            # ring orientation? reverse the vertex order so land stays on 
            # the right in FEMTIC's (X=north, Y=east) frame.
            coords = coords[::-1]
            n = len(coords)
            for i, (east_m, north_m) in enumerate(coords):
                north_km = (north_m - north0) / 1000.0
                east_km = (east_m - east0) / 1000.0
                end_flag = 0 if i < n - 1 else (1 if is_closed else -1)
                f.write(f"{north_km:.9f} {east_km:.9f} {end_flag} 0\n")


def _write_topo_files(topo_df, out_dir, model_center_m=(0.0, 0.0),
                      sea_level_m=0.0, land_name='topo_land.txt',
                      sea_name='topo_sea.txt', bounds_km=None):
    """Split topo into altitude / bathymetry files (X=north, Y=east).

    ``topo_df`` has ``east_m``/``north_m``/``z_m``; ``model_center_m`` is
    ``(north0_m, east0_m)``; ``bounds_km`` is
    ``(north_min, north_max, east_min, east_max)`` in centered km.
    """
    out_dir = pathlib.Path(out_dir)
    north0, east0 = model_center_m
    north_km = (topo_df['north_m'].to_numpy() - north0) / 1000.0
    east_km = (topo_df['east_m'].to_numpy() - east0) / 1000.0
    z_m = topo_df['z_m'].to_numpy()

    if bounds_km is not None:
        north_min, north_max, east_min, east_max = bounds_km
        in_box = (north_km >= north_min) & (north_km <= north_max) & \
                 (east_km >= east_min) & (east_km <= east_max)
        north_km = north_km[in_box]
        east_km = east_km[in_box]
        z_m = z_m[in_box]

    land_mask = z_m >= sea_level_m
    sea_mask = ~land_mask
    land_path = sea_path = None

    if land_mask.any():
        land_path = out_dir / land_name
        with open(land_path, 'w') as f:
            for nk, ek, z in zip(north_km[land_mask], east_km[land_mask],
                                 z_m[land_mask]):
                f.write(f"{nk:.9f} {ek:.9f} {z / 1000.0:.6e}\n")
    if sea_mask.any():
        sea_path = out_dir / sea_name
        with open(sea_path, 'w') as f:
            for nk, ek, z in zip(north_km[sea_mask], east_km[sea_mask],
                                 z_m[sea_mask]):
                f.write(f"{nk:.9f} {ek:.9f} {(-z) / 1000.0:.6e}\n")
    return land_path, sea_path


def _write_makeMtr_param(path, ellipsoids, center_km=(0.0, 0.0, 0.0),
                         rotation_deg=0.0):
    """Write ``makeMtr.param``. ``center_km`` is ``(north, east, z)``."""
    nx, ey, z = center_km
    with open(path, 'w') as f:
        f.write(f"{nx:.4f} {ey:.4f} {z:.4f}\n")
        f.write(f"{rotation_deg:.4f}\n")
        f.write(f"{len(ellipsoids)}\n")
        for e in ellipsoids:
            a, ln, fh, fvp, fvm = e
            f.write(f"{a:8.3f} {ln:8.3f} {fh:6.3f} {fvp:6.3f} {fvm:6.3f}\n")


def _station_altitude_m(mt_df_centered, topo_df=None, default_alt_m=0.0,
                        idw_k=3, idw_radius_m=1e6, idw_eps_m=1e-6):
    """Per-station altitude (m, positive up). ``topo_df`` has east_m/north_m/z_m."""
    stations = mt_df_centered['station'].drop_duplicates().tolist()
    out = {}
    if 'elev' in mt_df_centered.columns:
        for s in stations:
            row = mt_df_centered.loc[mt_df_centered['station'] == s].iloc[0]
            out[s] = float(row['elev'])
        return out
    if topo_df is None or len(topo_df) == 0:
        return {s: float(default_alt_m) for s in stations}
    te = topo_df['east_m'].to_numpy()
    tn = topo_df['north_m'].to_numpy()
    tz = topo_df['z_m'].to_numpy()
    for s in stations:
        row = mt_df_centered.loc[mt_df_centered['station'] == s].iloc[0]
        se = float(row['east'])
        sn = float(row['north'])
        d = np.hypot(te - se, tn - sn)
        in_radius = d <= idw_radius_m
        if not in_radius.any():
            out[s] = float(default_alt_m)
            continue
        d_in = d[in_radius]
        z_in = tz[in_radius]
        order = np.argsort(d_in)[:idw_k]
        d_sel = d_in[order]
        z_sel = z_in[order]
        w = 1.0 / (d_sel + idw_eps_m)
        out[s] = float(np.sum(w * z_sel) / np.sum(w))
    return out


def _write_obs_site(mt_df_centered, path, ellipsoids=None, topo_df=None,
                    default_alt_m=0.0):
    """Write ``obs_site.dat`` for makeMtr (X=north, Y=east, Z down)."""
    if ellipsoids is None:
        ellipsoids = [[0.5, 0.10, 0.3], [1.0, 0.20, 0.3], [1.5, 0.30, 0.3],
                      [2.0, 0.50, 0.3], [3.0, 1.00, 0.3], [5.0, 2.00, 0.3]]
    alts_m = _station_altitude_m(mt_df_centered, topo_df=topo_df,
                                 default_alt_m=default_alt_m)
    stations = mt_df_centered['station'].drop_duplicates().tolist()
    with open(path, 'w') as f:
        f.write(f"{len(stations)}\n")
        for s in stations:
            row = mt_df_centered.loc[mt_df_centered['station'] == s].iloc[0]
            north_km = float(row['north']) / 1000.0
            east_km = float(row['east']) / 1000.0
            z_km = -alts_m[s] / 1000.0
            f.write(f"{north_km:>14.9f} {east_km:>14.9f}  {z_km:>.6E}\n")
            f.write(f"{len(ellipsoids)}\n")
            for a, ln, fv in ellipsoids:
                f.write(f" {a} {ln:.2f} {fv}\n")


def _write_resistivity_attr(mt_df_centered, path, regions=None, ellipsoids=None,
                            center_km=(0.0, 0.0, 0.0), rotation_deg=0.0,
                            site_spheres=None, topo_df=None, default_alt_m=0.0):
    """Write ``resistivity_attr.dat``. ``center_km`` is ``(north, east, z)``."""
    if regions is None:
        regions = [
            {'attr': 10, 'rho': 1.0e12, 'repeat': -1, 'fixed': 1},
            {'attr': 20, 'rho': 0.3, 'repeat': -1, 'fixed': 1},
            {'attr': 30, 'rho': 100.0, 'repeat': 9, 'fixed': 0},
        ]
    if ellipsoids is None:
        ellipsoids = [
            [40.0, 2.0, 0.5, 0.7], [45.0, 3.0, 0.5, 0.7],
            [50.0, 5.0, 0.4, 0.7], [60.0, 10.0, 0.3, 0.6],
            [100.0, 100.0, 0.0, 0.5], [200.0, 200.0, 0.0, 0.3],
            [300.0, 300.0, 0.0, 0.2], [500.0, 500.0, 0.0, 0.1],
            [1000.0, 1000.0, 0.0, 0.0],
        ]
    if site_spheres is None:
        site_spheres = [[3.0, 2.0], [5.0, 3.0]]
    alts_m = _station_altitude_m(mt_df_centered, topo_df=topo_df,
                                 default_alt_m=default_alt_m)
    stations = mt_df_centered['station'].drop_duplicates().tolist()
    nx, ey, z = center_km
    with open(path, 'w') as f:
        f.write(f"{len(regions)}\n")
        for r in regions:
            f.write(f"{int(r['attr'])} {r['rho']:.3e} "
                    f"{int(r['repeat'])} {int(r['fixed'])}\n")
        f.write(f"{nx:.4f} {ey:.4f} {z:.4f}\n")
        f.write(f"{rotation_deg:.4f}\n")
        f.write(f"{len(ellipsoids)}\n")
        for e in ellipsoids:
            a, ln, fh, fv = e
            f.write(f"{a:8.3f} {ln:8.3f} {fh:6.3f} {fv:6.3f}\n")
        f.write(f"{len(stations)}\n")
        for s in stations:
            row = mt_df_centered.loc[mt_df_centered['station'] == s].iloc[0]
            north_km = float(row['north']) / 1000.0
            east_km = float(row['east']) / 1000.0
            z_km = -alts_m[s] / 1000.0
            f.write(f"{north_km:>14.9f} {east_km:>14.9f}  {z_km:>.6E}\n")
            f.write(f"{len(site_spheres)}\n")
            for r, ln in site_spheres:
                f.write(f"{r} {ln}\n")



class TetraMesh(FemticMesh):
    """Builder for the FEMTIC tetra-mesh pipeline's eight starting files.

    Coordinate convention
    ---------------------
    The model frame is **x = north, y = east, z = down** — the same as
    :mod:`hexmesh` and FEMTIC's files. So :attr:`model_center` is
    ``(north_m, east_m)``, :attr:`bounds_km` is north-first, and any
    ellipsoid ``center`` you pass (``control_center_km`` etc.) is
    ``(north_km, east_km, z_km)`` with ``rotation_deg`` measured in that
    same frame. (GIS/topo pipeline internally uses easting-as-X per shapely
    convention, but that never reaches this API.)

    :param mt_df: mtpy-v2 long DataFrame.
    :type mt_df: pandas.DataFrame
    :param start_res: Background resistivity (ohm-m) for skin-depth sizing,
        defaults to ``100.0``.
    :type start_res: float, optional
    :param center_mode: ``"bbox"`` or ``"mean"`` for the model origin,
        defaults to ``"bbox"``.
    :type center_mode: str, optional
    :param config: Any of the tetra-pipeline options (``topo_path``,
        ``coast_gdf``, ``derive_coast_from_topo``, ``padding_km``,
        ``bounds_km`` [north-first], ``control_ellipsoids``,
        ``control_center_km`` [north, east, z], ``resistivity_regions`` …).
    """

    # tetra-pipeline option names accepted in **config
    _CONFIG_KEYS = {
        'topo_path', 'topo_kind', 'topo_dst_crs', 'topo_target_spacing_m',
        'topo_xyz_kwargs', 'coast_gdf', 'derive_coast_from_topo', 'sea_level_m',
        'center_m', 'bounds_km', 'padding_km', 'start_res', 'depth_factor',
        'air_factor', 'control_ellipsoids', 'control_center_km',
        'control_rotation_deg', 'control_num_threads', 'interpolate',
        'obs_site_spheres', 'alt_filename', 'bathy_filename',
        'makemtr_ellipsoids', 'obs_site_ellipsoids', 'resistivity_regions',
        'resistivity_ellipsoids', 'resistivity_site_spheres',
        'coarse_spacing_m', 'dense_spacing_m', 'site_buffer_m', 'coast_buffer_m',
        'coast_simplify_tol_m', 'coast_grid_spacing_m',
        'coast_max_nodata_distance_m', 'coast_min_island_area_m2',
        'coast_min_lake_area_m2', 'coast_min_vertices', 'coast_distance_scale_m',
        'coast_distance_power', 'coast_min_ring_gap_m', 'land_on_right',
    }

    def __init__(self, mt_df: pd.DataFrame, *, start_res: float = 100.0,
                 center_mode: str = "bbox", **config):
        super().__init__(mt_df, start_res=start_res)
        self.center_mode = center_mode
        config.setdefault("start_res", start_res)
        unknown = set(config) - self._CONFIG_KEYS
        if unknown:
            warnings.warn(f"TetraMesh: unrecognised config keys ignored by "
                          f"write_inputs: {sorted(unknown)}", stacklevel=2)
        self.config = dict(config)

    @property
    def model_center(self) -> tuple:
        """Model origin ``(north_m, east_m)`` in source meters (x = north)."""
        c = self.config.get("center_m")
        if c is not None:
            return tuple(c)
        return _model_center(self.mt_df, mode=self.center_mode)

    @property
    def centered_dataframe(self) -> pd.DataFrame:
        """Copy of ``mt_df`` shifted so the model center is ``(0, 0)``."""
        df, _ = _recenter(self.mt_df, center=self.model_center,
                          mode=self.center_mode)
        return df

    @property
    def bounds_km(self) -> tuple:
        """Domain bounds ``(north_min, north_max, east_min, east_max, z_air, z_earth)`` (km).

        x = north, y = east — FEMTIC's order and the same axis sense as
        :class:`hexmesh.DeformableHexMesh`. Uses an explicit
        ``bounds_km`` from the config (also north-first) if given.
        """
        b = self.config.get("bounds_km")
        if b is not None:
            return tuple(b)
        return _auto_bounds_km(
            self.centered_dataframe,
            padding_km=self.config.get("padding_km", 500.0),
            start_res=self.config.get("start_res", self.start_res),
            depth_factor=self.config.get("depth_factor", 3.0),
            air_factor=self.config.get("air_factor", 1.0))

    # writers; wired the full "orchestrator" to work with an old function call

    def write_inputs(self, out_dir: PathLike) -> dict:
        """Write every tetra starting file into ``out_dir``.

        :param out_dir: Destination directory (created if missing).
        :type out_dir: str or pathlib.Path
        :return: ``{filename: path}`` for every file written.
        :rtype: dict
        """
        return self._write_all(out_dir, **self.config)

    def _write_all(self, out_dir,
                   topo_path=None, topo_kind='auto', topo_dst_crs=None,
                   topo_target_spacing_m=None, topo_xyz_kwargs=None,
                   coast_gdf=None, derive_coast_from_topo=False, sea_level_m=0.0,
                   center_m=None, bounds_km=None, padding_km=50.0,
                   start_res=100.0, depth_factor=3.0, air_factor=1.0,
                   control_ellipsoids=None, control_center_km=(0.0, 0.0, 0.0),
                   control_rotation_deg=0.0, control_num_threads=1,
                   interpolate=(10.0, 4, 1.0e-6), obs_site_spheres=None,
                   alt_filename='topo_land.txt', bathy_filename='topo_sea.txt',
                   makemtr_ellipsoids=None, obs_site_ellipsoids=None,
                   resistivity_regions=None, resistivity_ellipsoids=None,
                   resistivity_site_spheres=None,
                   coarse_spacing_m=4000.0, dense_spacing_m=400.0,
                   site_buffer_m=3000.0, coast_buffer_m=3000.0,
                   coast_simplify_tol_m=None, coast_grid_spacing_m=None,
                   coast_max_nodata_distance_m=None,
                   coast_min_island_area_m2=1.0e6, coast_min_lake_area_m2=None,
                   coast_min_vertices=None, coast_distance_scale_m=None,
                   coast_distance_power=2.0, coast_min_ring_gap_m=500.0,
                   land_on_right=True):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        have_gpd = _has_geopandas()

        # Recenter MT data (center = (north0, east0)).
        mt_df_c, center = _recenter(self.mt_df, center=center_m,
                                    mode=self.center_mode)
        north0, east0 = center

        if bounds_km is None:
            bounds_km = _auto_bounds_km(mt_df_c, padding_km=padding_km,
                                        start_res=start_res,
                                        depth_factor=depth_factor,
                                        air_factor=air_factor)
        north_min, north_max, east_min, east_max, _, _ = bounds_km
        written = {}

        if control_ellipsoids is None:
            control_ellipsoids = [
                [40.0, 1.0, 0.5, 0.5, 0.7], [60.0, 5.0, 0.3, 0.3, 0.5],
                [100.0, 10.0, 0.2, 0.1, 0.3], [200.0, 20.0, 0.0, 0.0, 0.0],
                [300.0, 30.0, 0.0, 0.0, 0.0], [500.0, 50.0, 0.0, 0.0, 0.0]]
        if makemtr_ellipsoids is None:
            makemtr_ellipsoids = [
                [40.0, 1.0, 0.5, 0.7, 0.9], [45.0, 1.5, 0.5, 0.5, 0.7],
                [50.0, 3.0, 0.5, 0.4, 0.7], [60.0, 5.0, 0.3, 0.3, 0.5],
                [80.0, 8.0, 0.2, 0.1, 0.3], [100.0, 10.0, 0.0, 0.0, 0.0],
                [200.0, 20.0, 0.0, 0.0, 0.0], [300.0, 30.0, 0.0, 0.0, 0.0],
                [400.0, 40.0, 0.0, 0.0, 0.0], [500.0, 50.0, 0.0, 0.0, 0.0]]

        topo_df = None
        alt_present = bathy_present = False
        if topo_path is not None:
            if topo_kind == 'auto':
                topo_kind = _detect_topo_kind(topo_path)
            # Source-CRS clip box: MT extent + padding, in (east, north) meters.
            bounds_src = (east_min * 1000.0 + east0, north_min * 1000.0 + north0,
                          east_max * 1000.0 + east0, north_max * 1000.0 + north0)
            if topo_kind == 'xyz':
                topo_full = _load_topo(topo_path, kind='xyz',
                                       dst_crs=topo_dst_crs,
                                       target_spacing_m=topo_target_spacing_m,
                                       bounds_m=bounds_src,
                                       **dict(topo_xyz_kwargs or {}))
            elif topo_kind == 'raster':
                topo_full = _load_topo(topo_path, kind='raster',
                                       dst_crs=topo_dst_crs,
                                       target_spacing_m=topo_target_spacing_m,
                                       bounds_m=bounds_src)
            else:
                raise ValueError(f"Unknown topography kind: {topo_kind!r}")

            if derive_coast_from_topo and coast_gdf is None and have_gpd \
                    and len(topo_full) > 0:
                if topo_kind == 'xyz':
                    coast_grid = coast_grid_spacing_m if coast_grid_spacing_m \
                        is not None else 1000.0
                else:
                    coast_grid = coast_grid_spacing_m
                coast_gdf = _derive_coast_gdf_from_df(
                    topo_full, sea_level=sea_level_m, dst_crs=topo_dst_crs,
                    grid_spacing_m=coast_grid,
                    max_nodata_distance_m=coast_max_nodata_distance_m,
                    simplify_tol_m=None,
                    min_island_area_m2=coast_min_island_area_m2,
                    min_lake_area_m2=coast_min_lake_area_m2,
                    min_vertices=coast_min_vertices,
                    distance_scale_m=coast_distance_scale_m,
                    distance_power=coast_distance_power,
                    # distance from MT origin in source (easting, northing)
                    distance_center=(east0, north0))

            centered_coast_gdf = None
            if coast_gdf is not None and len(coast_gdf) > 0 and have_gpd:
                from shapely.affinity import translate
                centered_coast_gdf = coast_gdf.copy()
                # geometry is (easting, northing): xoff=-east0, yoff=-north0
                centered_coast_gdf['geometry'] = centered_coast_gdf.geometry.apply(
                    lambda g: translate(g, -east0, -north0))

            # Sample grid bounds in centered (east, north) meters.
            dom_bounds = (east_min * 1000.0, north_min * 1000.0,
                          east_max * 1000.0, north_max * 1000.0)
            if len(topo_full) > 0:
                topo_extent = (topo_full['east_m'].min() - east0,
                               topo_full['north_m'].min() - north0,
                               topo_full['east_m'].max() - east0,
                               topo_full['north_m'].max() - north0)
                grid_bounds = (max(dom_bounds[0], topo_extent[0]),
                               max(dom_bounds[1], topo_extent[1]),
                               min(dom_bounds[2], topo_extent[2]),
                               min(dom_bounds[3], topo_extent[3]))
            else:
                grid_bounds = dom_bounds

            grid = _create_topo_point_grid(
                mt_df_c, coast_gdf=centered_coast_gdf,
                coarse_spacing_m=coarse_spacing_m,
                dense_spacing_m=dense_spacing_m, site_buffer_m=site_buffer_m,
                coast_buffer_m=coast_buffer_m, bounds_m=grid_bounds)
            grid_e = grid['east_m'].to_numpy()
            grid_n = grid['north_m'].to_numpy()

            from scipy.spatial import cKDTree
            if len(topo_full) > 0:
                tree = cKDTree(np.column_stack([topo_full['east_m'].to_numpy(),
                                                topo_full['north_m'].to_numpy()]))
                query_e = grid_e + east0
                query_n = grid_n + north0
                _, idx = tree.query(np.column_stack([query_e, query_n]), k=1)
                topo_df = pd.DataFrame({
                    'east_m': query_e, 'north_m': query_n,
                    'z_m': topo_full['z_m'].to_numpy()[idx]})

        # analysis_domain.dat
        p_ad = out_dir / 'analysis_domain.dat'
        _write_analysis_domain(p_ad, bounds_km)
        written['analysis_domain.dat'] = p_ad

        # altitude / bathymetry
        if topo_df is not None and len(topo_df) > 0:
            land_path, sea_path = _write_topo_files(
                topo_df, out_dir, model_center_m=center,
                sea_level_m=sea_level_m, land_name=alt_filename,
                sea_name=bathy_filename,
                bounds_km=(north_min, north_max, east_min, east_max))
            alt_present = land_path is not None
            bathy_present = sea_path is not None
            if alt_present:
                written[alt_filename] = land_path
            if bathy_present:
                written[bathy_filename] = sea_path

        # control.dat; this collides with the control.dat for inversions, but Yoshi's code is wired to look for this filename.
        p_ctrl = out_dir / 'control.dat'
        _write_control(p_ctrl, ellipsoids=control_ellipsoids,
                       center_km=control_center_km,
                       rotation_deg=control_rotation_deg,
                       num_threads=control_num_threads, surf_mesh=True,
                       altitude_file=(alt_filename if alt_present else None),
                       sea_depth_file=(bathy_filename if bathy_present else None),
                       interpolate=interpolate)
        written['control.dat'] = p_ctrl

        # coast_line.dat
        p_coast = out_dir / 'coast_line.dat'
        if coast_gdf is not None and len(coast_gdf) > 0:
            _write_coast_line(coast_gdf, p_coast, model_center_m=center,
                              bounds_km=(north_min, north_max, east_min, east_max),
                              simplify_tol_m=coast_simplify_tol_m,
                              land_on_right=land_on_right,
                              min_ring_gap_m=coast_min_ring_gap_m)
        else:
            with open(p_coast, 'w') as f:
                f.write("0\n")
        written['coast_line.dat'] = p_coast

        # observing_site.dat; another case of some potentially-confusing naming that nonetheless seems to be required.
        p_obs1 = out_dir / 'observing_site.dat'
        _write_observing_site(mt_df_c, p_obs1, spheres=obs_site_spheres)
        written['observing_site.dat'] = p_obs1

        # makeMtr.param
        p_mm = out_dir / 'makeMtr.param'
        _write_makeMtr_param(p_mm, ellipsoids=makemtr_ellipsoids,
                             center_km=control_center_km,
                             rotation_deg=control_rotation_deg)
        written['makeMtr.param'] = p_mm

        # obs_site.dat
        p_obs2 = out_dir / 'obs_site.dat'
        _write_obs_site(mt_df_c, p_obs2, ellipsoids=obs_site_ellipsoids,
                        topo_df=topo_df)
        written['obs_site.dat'] = p_obs2

        # resistivity_attr.dat
        p_rho = out_dir / 'resistivity_attr.dat'
        _write_resistivity_attr(mt_df_c, p_rho, regions=resistivity_regions,
                                ellipsoids=resistivity_ellipsoids,
                                center_km=control_center_km,
                                rotation_deg=control_rotation_deg,
                                site_spheres=resistivity_site_spheres,
                                topo_df=topo_df)
        written['resistivity_attr.dat'] = p_rho

        self.logger.info(
            f"Wrote {len(written)} FEMTIC tetra input file(s) to: {out_dir}"
        )
        return written
