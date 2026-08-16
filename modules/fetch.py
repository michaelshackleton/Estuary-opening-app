"""
Fetches Landsat/Sentinel-2 scenes from Digital Earth Australia's STAC
catalogue for an on-the-fly, user-drawn region of interest, and runs the
open/closed connectivity analysis (see connectivity.py) on each scene.

Reuses the STACDataManager / RSDataProductManager classes originally from
this repo's src/rs_data.py and src/rs_processing.py, vendored verbatim into
this app's vendor/ folder (along with their own dependencies - utils.py and
rsderiv/sharpen.py, rsderiv/sen1.py) so that this app folder is fully
self-contained and can be copied/moved (or deployed to Streamlit Community
Cloud) without needing the rest of the rs-utils-main repository alongside
it. The one thing this module deliberately avoids is
STACDataManager.stac_ds_to_product()'s `.rio.to_raster()` write path, which
throws "Only 2D and 3D data arrays supported" on this project's multiband
datasets - scenes are instead written to GeoTIFF here directly
(write_scene_geotiff), band-by-band, which sidesteps whatever is tripping
up that path.

STACRSRegionManager (in rs_data.py) only knows how to load a region's
polygon from a shapefile/geopackage path via a region parameter JSON file.
Since this app defines regions on the fly by drawing on a map, rather than
editing that class, InMemoryRegionManager below duck-types the same
interface (get_polygon / get_region_extent) directly from a GeoDataFrame.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Callable, Optional

import numpy as np
import pandas as pd
import rasterio

_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_MODULES_DIR)
_VENDOR_DIR = os.path.join(_APP_DIR, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from rs_data import STACDataManager  # noqa: E402
from rs_processing import RSDataProductManager  # noqa: E402

from . import connectivity  # noqa: E402
from .region import SiteLayers  # noqa: E402

ProgressCB = Optional[Callable[[int, int, str], None]]

DEFAULT_PRODUCTS = ("landsat_full", "sentinel_full")

# "Full coverage" is treated as >= 99.9% rather than exactly 100% to allow
# for floating-point noise in the polygon-intersection area calculation -
# a genuinely fully-covering tile can come out as 0.999999... rather than
# an exact 1.0.
FULL_COVERAGE_THRESHOLD = 0.999

# Human-readable sensor label per product code, used to keep the 'sensor'
# field consistent between successful rows (labelled by connectivity.py's
# auto-detection) and error rows (which never reach that auto-detection).
PRODUCT_SENSOR_LABEL = {"landsat_full": "landsat", "sentinel_full": "sentinel2"}
SENSOR_PRODUCT = {v: k for k, v in PRODUCT_SENSOR_LABEL.items()}


class InMemoryRegionManager:
    """Duck-types the subset of STACRSRegionManager's interface that
    STACDataManager / RSDataProductManager actually use, backed by an
    in-memory GeoDataFrame instead of a file on disk."""

    def __init__(self, roi_gdf, region_name: str = "site"):
        self.region_poly = roi_gdf
        self.region_name = region_name
        self.region_code = region_name
        self.sub_region_col = None

    def get_polygon(self, sub_region=None, target_crs=None):
        gdf = self.region_poly.to_crs(target_crs) if target_crs else self.region_poly
        return gdf.iloc[0].geometry

    def get_region_extent(self, sub_region=None, target_crs=None):
        gdf = self.region_poly.to_crs(target_crs) if target_crs else self.region_poly
        return gdf.total_bounds


def write_scene_geotiff(bands: dict, transform, crs, out_path: str):
    band_names = list(bands.keys())
    arr = np.stack([np.asarray(bands[b]) for b in band_names], axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def read_scene_geotiff(path: str):
    with rasterio.open(path) as src:
        bands = {}
        for i in range(1, src.count + 1):
            name = src.descriptions[i - 1] or f"band_{i}"
            bands[name] = src.read(i)
        transform = src.transform
        crs = src.crs
    return bands, transform, crs


def _scene_cache_path(cache_dir: str, site_name: str, product_code: str, date: pd.Timestamp) -> str:
    fname = f"{product_code}_{date.strftime('%Y-%m-%d')}.tif"
    return os.path.join(cache_dir, site_name, product_code, fname)


# --------------------------------------------------------------------------
# Temporal clear-sky reference caching
# --------------------------------------------------------------------------
#
# run_site_analysis() builds a per-pixel clear-sky reference (see
# connectivity.build_clear_sky_reference) "for free" from the full scene
# time series it already loads into memory for a given site+product - no
# extra fetching needed. This is cached to disk (one reference per
# site/product, overwritten on each run that has temporal-anomaly detection
# enabled) so that fetch_single_scene()'s on-demand single-scene preview can
# reuse the same reference for its diagnostic overlay, without needing to
# re-fetch a whole site history just to preview one date.
#
# On Streamlit Community Cloud, `cache_dir` sits on the container's local
# ephemeral disk - it works fine within a running session, but is wiped on
# every redeploy/restart (the container filesystem isn't persistent storage,
# unlike a desktop install where data_cache/ survives indefinitely). See
# this app's README for what that means in practice.

def _reference_cache_paths(cache_dir: str, site_name: str, product_code: str) -> tuple[str, str]:
    base = os.path.join(cache_dir, site_name, product_code, "clear_sky_reference")
    return base + ".npy", base + ".json"


def save_temporal_reference_cache(
    cache_dir: str, site_name: str, product_code: str, reference: np.ndarray,
    percentile: float, min_clear_obs: int, start_date: str, end_date: str,
) -> None:
    npy_path, json_path = _reference_cache_paths(cache_dir, site_name, product_code)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, reference)
    meta = {
        "shape": list(reference.shape),
        "percentile": percentile,
        "min_clear_obs": min_clear_obs,
        "start_date": start_date,
        "end_date": end_date,
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_cached_temporal_reference(
    cache_dir: Optional[str], site_name: str, product_code: str, expected_shape: Optional[tuple] = None
) -> Optional[np.ndarray]:
    """Loads a previously-cached clear-sky reference for this site/product,
    or None if there isn't one (temporal-anomaly detection has never been
    run for this site) or `expected_shape` is given and doesn't match (the
    cached reference is from a different grid - e.g. the ROI was redrawn -
    so it's silently ignored rather than risk misaligned masking)."""
    if not cache_dir:
        return None
    npy_path, json_path = _reference_cache_paths(cache_dir, site_name, product_code)
    if not (os.path.exists(npy_path) and os.path.exists(json_path)):
        return None
    try:
        reference = np.load(npy_path)
    except Exception:
        return None
    if expected_shape is not None and tuple(reference.shape) != tuple(expected_shape):
        return None
    return reference


def run_site_analysis(
    site: SiteLayers,
    start_date: str,
    end_date: str,
    max_cloud: float,
    products_json_path: str,
    cache_dir: Optional[str] = None,
    cache_rasters: bool = False,
    products=DEFAULT_PRODUCTS,
    min_roi_coverage: float = FULL_COVERAGE_THRESHOLD,
    progress_cb: ProgressCB = None,
    cloud_buffer_px: Optional[int | dict] = None,
    enable_temporal_anomaly: bool = False,
    temporal_anomaly_threshold: float = connectivity.DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN,
    temporal_anomaly_percentile: float = 10.0,
    temporal_anomaly_min_obs: int = 3,
) -> list[dict]:
    """Fetches every available scene over `site`'s ROI in [start_date,
    end_date] with cloud cover <= max_cloud, runs the open/closed
    connectivity analysis on each, and returns a list of per-scene result
    rows (dicts) - one per record.date/sensor - ready for
    aggregate.build_results_df().

    `min_roi_coverage` (0-1) drops any scene whose tile doesn't cover at
    least that fraction of the drawn ROI polygon - e.g. dates where the ROI
    straddles two tiles collected on different days, or sits right at the
    edge of a swath. Defaults to requiring (near-)full coverage, since a
    partially-covering tile leaves part of the ROI as nodata, which usually
    just produces an indeterminate result rather than a useful one - so
    filtering these out up front both avoids that noise and cuts down the
    number of scenes that need fetching/analysing. Pass 0 to disable this
    filter and keep every scene regardless of coverage.

    The connectivity analysis itself only ever needs each scene in memory
    momentarily - it never reads back from a cached file - so caching
    rasters to disk here is purely an optional convenience, not something
    the analysis depends on. `cache_rasters=False` (the default) skips
    writing any GeoTIFFs at all during this run, which is what keeps
    `data_cache/` from filling up with every scene in a multi-decade run
    when most of them will never be looked at again. Set `cache_rasters=
    True` if you specifically want every scene from this run cached ahead
    of time for fast preview clicking afterwards (uses much more disk).
    Regardless of this setting, `fetch_single_scene()` can fetch any one
    scene on demand later - which is what the app's scene preview uses by
    default instead of relying on a pre-populated cache.

    `cache_dir` is only actually used as a location to write to when
    `cache_rasters=True` (or via `fetch_single_scene()` below) - if omitted,
    a temporary directory is used instead (STACDataManager requires some
    root directory to exist, but writes nothing into it unless asked to).

    `cloud_buffer_px` is passed straight through to
    connectivity.process_scene() for every scene - see that function's
    docstring (int, {"sentinel2": n, "landsat": n} dict, or None for
    connectivity.DEFAULT_CLOUD_BUFFER_PX).

    `enable_temporal_anomaly` builds a per-pixel clear-sky reference (see
    connectivity.build_clear_sky_reference) from this run's own scene time
    series for each product, and applies connectivity's temporal-anomaly
    check to every scene using it - complementary to the cloud-edge buffer,
    since it can catch thin/dappled cloud fmask's own algorithm never
    flagged as cloud at all (which the buffer, only ever dilating outward
    from cells fmask DID flag, cannot). Off by default since it needs
    `nbart_blue` in the product's bands and adds a bit of per-run compute;
    when `cache_dir` is given, the built reference is also cached to disk
    (see save_temporal_reference_cache/load_cached_temporal_reference) so
    fetch_single_scene()'s preview path can reuse it later. `_percentile`/
    `_min_obs` control how the reference itself is built - see
    connectivity.build_clear_sky_reference's docstring.
    """
    problems = site.validate()
    if problems:
        raise ValueError("Site layers are not valid: " + "; ".join(problems))

    target_crs = site.target_crs()
    inside = site.inside_line(target_crs)
    outside = site.outside_line(target_crs)
    structures = site.structures_reprojected(target_crs)

    region_mgr = InMemoryRegionManager(site.roi, region_name=site.name)
    work_dir = cache_dir or tempfile.mkdtemp(prefix="estuary_cache_")
    stac_mgr = STACDataManager(root_dir=work_dir)

    records: list[dict] = []

    for product_code in products:
        product_mgr = RSDataProductManager(product_code, product_param_path=products_json_path)

        if progress_cb:
            progress_cb(0, 1, f"Searching {product_code} scenes...")
        items = stac_mgr.get_stac_items(start_date, end_date, product_mgr, region_mgr)
        items = stac_mgr.filter_stac_items_eocloud(items, max_cloud)
        if not items:
            continue
        if min_roi_coverage > 0:
            items = stac_mgr.filter_stac_items_region_overlap(
                items, min_roi_coverage, region_mgr, target_crs=target_crs
            )
        if not items:
            continue

        xr_ds = stac_mgr.stac_items_to_xrdataset(items, product_mgr, region_mgr, target_crs=target_crs)
        # make_multiband is @dask.delayed; compute=True resolves it to an
        # in-memory xarray.Dataset already clipped to the ROI polygon.
        processed = product_mgr.make_product_dask(xr_ds, region_mgr, compute=True)

        if "time" not in processed.dims:
            processed = processed.expand_dims("time")
        time_values = np.atleast_1d(processed.time.values)
        n_scenes = len(time_values)

        # Build the temporal clear-sky reference "for free" from the full
        # time series already loaded above, before scoring any individual
        # scene against it - see connectivity.build_clear_sky_reference and
        # the enable_temporal_anomaly docstring above.
        clear_sky_reference = None
        if enable_temporal_anomaly:
            if "nbart_blue" in processed.data_vars:
                if progress_cb:
                    progress_cb(0, 1, f"Building clear-sky reference for {product_code}...")
                blue_stack = np.asarray(processed["nbart_blue"].values)
                fmask_stack = np.asarray(processed["oa_fmask"].values)
                clear_sky_reference = connectivity.build_clear_sky_reference(
                    blue_stack, fmask_stack,
                    percentile=temporal_anomaly_percentile, min_clear_obs=temporal_anomaly_min_obs,
                )
                if cache_dir:
                    try:
                        save_temporal_reference_cache(
                            cache_dir, site.name, product_code, clear_sky_reference,
                            percentile=temporal_anomaly_percentile, min_clear_obs=temporal_anomaly_min_obs,
                            start_date=start_date, end_date=end_date,
                        )
                    except Exception as e:  # caching failures shouldn't stop analysis
                        print(f"Warning: could not cache clear-sky reference for {product_code}: {e}")
            else:
                print(f"Warning: temporal-anomaly detection requested but 'nbart_blue' not in {product_code}'s bands - skipping for this product.")

        # Keyed by calendar date so that two STAC items sharing the same date
        # (e.g. adjacent tiles from one overpass, captured seconds apart with
        # distinct timestamps) don't both become separate rows - only the
        # better-covered one is kept per date. Without this, a downstream
        # "prefer Sentinel on shared dates" dedupe step only distinguishes by
        # sensor, not by which of two same-sensor rows actually has data, so
        # it could silently keep an all-no-data duplicate instead of the good
        # one depending on row order.
        product_records: dict = {}

        for i, t in enumerate(time_values):
            if progress_cb:
                progress_cb(i, n_scenes, f"Analysing {product_code}: scene {i + 1}/{n_scenes}")

            scene = processed.isel(time=i)
            bands = {var: np.asarray(scene[var].values) for var in scene.data_vars}
            transform = scene.rio.transform()
            date = pd.Timestamp(t).normalize()

            cache_path = None
            if cache_rasters and cache_dir:
                cache_path = _scene_cache_path(cache_dir, site.name, product_code, date)
                if not os.path.exists(cache_path):
                    try:
                        write_scene_geotiff(bands, transform, f"EPSG:{target_crs}", cache_path)
                    except Exception as e:  # caching failures shouldn't stop analysis
                        print(f"Warning: could not cache scene {cache_path}: {e}")

            try:
                result = connectivity.process_scene(
                    bands, transform, inside, outside, structures, cloud_buffer_px=cloud_buffer_px,
                    clear_sky_reference=clear_sky_reference, temporal_anomaly_threshold=temporal_anomaly_threshold,
                )
                record = dict(
                    date=date,
                    sensor=result.sensor,
                    status=result.status,
                    status_ndwi=result.status_ndwi,
                    status_fmask=result.status_fmask,
                    gap_ndwi=result.gap_ndwi,
                    gap_fmask=result.gap_fmask,
                    reason_ndwi=result.reason_ndwi,
                    reason_fmask=result.reason_fmask,
                    n_nodata_cells=result.n_nodata_cells,
                    pct_cloud=result.pct_cloud,
                    pct_cloud_shadow=result.pct_cloud_shadow,
                    cloud_buffer_px=result.cloud_buffer_px,
                    pct_temporal_anomaly=result.pct_temporal_anomaly,
                    temporal_reference_coverage_pct=result.temporal_reference_coverage_pct,
                    cache_path=cache_path,
                    error=None,
                )
                is_error, nodata = False, result.n_nodata_cells
            except Exception as e:
                record = dict(
                    date=date,
                    sensor=PRODUCT_SENSOR_LABEL.get(product_code, product_code),
                    status="error",
                    status_ndwi=None,
                    status_fmask=None,
                    gap_ndwi=None,
                    gap_fmask=None,
                    reason_ndwi=None,
                    reason_fmask=None,
                    n_nodata_cells=None,
                    pct_cloud=None,
                    pct_cloud_shadow=None,
                    cloud_buffer_px=None,
                    pct_temporal_anomaly=None,
                    temporal_reference_coverage_pct=None,
                    cache_path=cache_path,
                    error=str(e),
                )
                is_error, nodata = True, None

            candidate_key = (is_error, nodata if nodata is not None else float("inf"))
            existing = product_records.get(date)
            if existing is None or candidate_key < existing[0]:
                product_records[date] = (candidate_key, record)

        records.extend(record for _, record in product_records.values())

    return records


def fetch_single_scene(
    site: SiteLayers,
    target_date,
    sensor: str,
    products_json_path: str,
    cache_dir: Optional[str] = None,
):
    """Fetches just the one scene for `target_date` and `sensor`
    ("landsat" or "sentinel2") - used by the app's scene preview to pull a
    raster on demand when a point on the results plot is clicked, rather
    than requiring every scene from a run to have already been cached to
    disk. If `cache_dir` is given, the fetched scene is written there
    (reusing the same cache_path convention as run_site_analysis, so a
    scene cached during a run with cache_rasters=True is found and reused
    rather than re-fetched) and the cache path is returned alongside the
    bands, so a repeat click on the same point is instant.

    Returns (bands, transform, crs, cache_path_or_None, diagnostics,
    clear_sky_reference). `diagnostics` is a dict with `n_items_found`,
    `items` (per-item id/datetime/collection/cloud_cover/region_overlap_pct,
    or None if it couldn't be computed) and `valid_pixel_frac` (fraction of
    the clipped ROI that's not oa_fmask no-data - a quick way to tell
    whether a near-blank preview is a real coverage gap versus something
    else going wrong upstream). `clear_sky_reference` is whatever
    load_cached_temporal_reference() finds for this site/product if
    `cache_dir` is given and a previous run_site_analysis() call with
    enable_temporal_anomaly=True has been done for this site - None
    otherwise (this function never builds one itself, since that needs a
    whole site time series, not just one scene).
    """
    if sensor not in SENSOR_PRODUCT:
        raise ValueError(f"Unknown sensor '{sensor}' - expected one of {list(SENSOR_PRODUCT)}")
    product_code = SENSOR_PRODUCT[sensor]

    target_crs = site.target_crs()
    date = pd.Timestamp(target_date).normalize()

    def _valid_pixel_frac(bands):
        if "oa_fmask" not in bands:
            return None
        return float(np.mean(np.isin(bands["oa_fmask"], [1, 2, 3, 4, 5])))

    cache_path = _scene_cache_path(cache_dir, site.name, product_code, date) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        bands, transform, crs = read_scene_geotiff(cache_path)
        diagnostics = {
            "source": "cache",
            "n_items_found": None,
            "items": None,
            "valid_pixel_frac": _valid_pixel_frac(bands),
        }
        clear_sky_reference = load_cached_temporal_reference(
            cache_dir, site.name, product_code,
            expected_shape=bands["nbart_blue"].shape if "nbart_blue" in bands else None,
        )
        return bands, transform, crs, cache_path, diagnostics, clear_sky_reference

    region_mgr = InMemoryRegionManager(site.roi, region_name=site.name)
    work_dir = cache_dir or tempfile.mkdtemp(prefix="estuary_cache_")
    stac_mgr = STACDataManager(root_dir=work_dir)
    product_mgr = RSDataProductManager(product_code, product_param_path=products_json_path)

    date_str = date.strftime("%Y-%m-%d")
    # Search a window around the target date rather than start==end (a
    # zero-width interval some STAC servers handle inconsistently right at
    # the day boundary) - mirrors run_site_analysis's wide-range query more
    # closely than a same-day-only search does.
    window_start = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    window_end = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    items = stac_mgr.get_stac_items(window_start, window_end, product_mgr, region_mgr)
    if not items:
        raise ValueError(f"No {sensor} scene found for {date_str} over this region.")

    # Diagnostic info about exactly what was found, independent of whether
    # the resulting raster ends up looking right - so a near-blank preview
    # can be told apart from "genuinely low coverage on this date" versus
    # something going wrong elsewhere in the pipeline.
    item_info = []
    try:
        item_gdf = stac_mgr.stac_items_to_gdf(items, region_manager=region_mgr, target_crs=target_crs)
        for item, (_, row) in zip(items, item_gdf.iterrows()):
            item_info.append(
                {
                    "id": item.id,
                    "collection": item.collection_id,
                    "datetime": item.properties.get("datetime"),
                    "cloud_cover": item.properties.get("eo:cloud_cover"),
                    "region_overlap_pct": round(float(row["region_overlap"]) * 100, 1),
                }
            )
    except Exception as e:  # diagnostics are best-effort, never fatal
        item_info = [{"error": str(e)}]

    xr_ds = stac_mgr.stac_items_to_xrdataset(items, product_mgr, region_mgr, target_crs=target_crs)
    processed = product_mgr.make_product_dask(xr_ds, region_mgr, compute=True)
    if "time" not in processed.dims:
        processed = processed.expand_dims("time")

    time_values = [pd.Timestamp(t).normalize() for t in processed.time.values]
    matches = [i for i, t in enumerate(time_values) if t == date]
    if not matches:
        # Don't silently fall back to whatever the first time slice happens to
        # be - on a nearby date that's a different (and possibly barely
        # overlapping) scene, which used to render as a near-blank/all-nodata
        # preview with no indication anything was wrong. Surface it instead.
        found = ", ".join(sorted({t.strftime("%Y-%m-%d") for t in time_values})) or "none"
        raise ValueError(
            f"Found {sensor} data near {date_str}, but not exactly on that date after "
            f"reprojecting (dates found: {found}). This can happen right at a UTC/local "
            "date boundary - the scene may need a small tolerance fix; please report this "
            "date if you see it again."
        )

    # A single calendar date can have more than one STAC item - e.g. two
    # adjacent Sentinel-2 tiles from the same overpass, captured seconds
    # apart, each getting its own distinct timestamp/time-slice rather than
    # being merged into one. Picking "the first match" arbitrarily used to
    # mean a 50/50 chance of landing on a tile with ~0% overlap with the ROI
    # (all no-data after clipping) instead of the one that actually covers
    # it. Evaluate every same-date candidate and keep whichever has the most
    # valid (non-no-data) pixels.
    best_idx, best_bands, best_transform, best_valid_frac = None, None, None, -1.0
    for cand_idx in matches:
        cand_scene = processed.isel(time=cand_idx)
        cand_bands = {var: np.asarray(cand_scene[var].values) for var in cand_scene.data_vars}
        vf = _valid_pixel_frac(cand_bands) or 0.0
        if vf > best_valid_frac:
            best_idx, best_bands, best_transform, best_valid_frac = (
                cand_idx, cand_bands, cand_scene.rio.transform(), vf,
            )

    idx, bands, transform = best_idx, best_bands, best_transform

    if cache_path:
        try:
            write_scene_geotiff(bands, transform, f"EPSG:{target_crs}", cache_path)
        except Exception as e:
            print(f"Warning: could not cache scene {cache_path}: {e}")

    diagnostics = {
        "source": "fetched",
        "n_items_found": len(items),
        "n_same_date_candidates": len(matches),
        "items": item_info,
        "valid_pixel_frac": _valid_pixel_frac(bands),
    }
    clear_sky_reference = load_cached_temporal_reference(
        cache_dir, site.name, product_code,
        expected_shape=bands["nbart_blue"].shape if "nbart_blue" in bands else None,
    )
    return bands, transform, rasterio.crs.CRS.from_epsg(target_crs), cache_path, diagnostics, clear_sky_reference


def cache_size_bytes(cache_dir: str) -> int:
    """Total size in bytes of everything under cache_dir, or 0 if it
    doesn't exist yet."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    total = 0
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def clear_cache(cache_dir: str):
    """Deletes everything under cache_dir (but not the folder itself), so
    a subsequent scene preview click just re-fetches from DEA. Does not
    affect any already-computed results - those live in the app's session
    state / exported CSV, never in this cache."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass
