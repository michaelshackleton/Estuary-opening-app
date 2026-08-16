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
(write_scene_geotiff_bytes), band-by-band, into an in-memory buffer, which
sidesteps whatever is tripping up that path.

NO SERVER-SIDE RASTER CACHING. This module deliberately never writes
fetched scenes to disk. On a shared hosting platform like Streamlit
Community Cloud, the whole app runs as one process shared by every visitor
- st.session_state is isolated per browser session, but a file written to
the server's own filesystem is not, so a disk cache keyed only by
site-name/sensor/date (as an earlier version of this module had) risks one
user silently loading another user's cached raster if they happen to pick
the same site name and an overlapping date. Every scene this module
fetches is handed back to the caller in memory only; app.py is responsible
for keeping (at most) the single currently-previewed scene in
st.session_state, and for offering it to the user as a download via
write_scene_geotiff_bytes() if they want a local copy - never onto the
server's disk.

STACRSRegionManager (in rs_data.py) only knows how to load a region's
polygon from a shapefile/geopackage path via a region parameter JSON file.
Since this app defines regions on the fly by drawing on a map, rather than
editing that class, InMemoryRegionManager below duck-types the same
interface (get_polygon / get_region_extent) directly from a GeoDataFrame.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Callable, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile

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


def write_scene_geotiff_bytes(bands: dict, transform, crs) -> bytes:
    """Encodes `bands` as a multi-band GeoTIFF entirely in memory (via
    rasterio's MemoryFile - no path on disk is ever touched) and returns
    the raw bytes, ready for st.download_button's `data=` argument. This is
    how the app lets a user save a scene to their own computer without the
    server ever writing it to its own filesystem."""
    band_names = list(bands.keys())
    arr = np.stack([np.asarray(bands[b]) for b in band_names], axis=0)
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
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(arr)
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
        return memfile.read()


def _stac_work_dir() -> str:
    """A fresh, private temp directory for STACDataManager's root_dir
    requirement (it insists some directory exists, but this app's fetch
    path never actually writes rasters into it - see module docstring).
    A new tempfile.mkdtemp() per call, rather than one fixed shared path,
    means nothing here is predictable/collidable between concurrent
    users even though nothing meaningful is written there anyway."""
    return tempfile.mkdtemp(prefix="estuary_stac_")


def run_site_analysis(
    site: SiteLayers,
    start_date: str,
    end_date: str,
    max_cloud: float,
    products_json_path: str,
    products=DEFAULT_PRODUCTS,
    min_roi_coverage: float = FULL_COVERAGE_THRESHOLD,
    progress_cb: ProgressCB = None,
    cloud_buffer_px: Optional[int | dict] = None,
    enable_temporal_anomaly: bool = False,
    temporal_anomaly_threshold: float = connectivity.DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN,
    temporal_anomaly_percentile: float = 10.0,
    temporal_anomaly_min_obs: int = 3,
) -> tuple[list[dict], dict[str, Optional[np.ndarray]]]:
    """Fetches every available scene over `site`'s ROI in [start_date,
    end_date] with cloud cover <= max_cloud, runs the open/closed
    connectivity analysis on each, and returns
    (records, clear_sky_references).

    `records` is a list of per-scene result rows (dicts) - one per
    record.date/sensor - ready for aggregate.build_results_df(). No raster
    is ever written to disk here; each scene is held in memory only long
    enough to run the connectivity check, then discarded - the analysis
    itself never needs a raster again after that.

    `clear_sky_references` is `{product_code: 2D numpy array or None}` -
    the per-pixel clear-sky brightness reference built from this run's own
    scene time series (see connectivity.build_clear_sky_reference), one per
    product, if `enable_temporal_anomaly=True` (None per product
    otherwise, or if that product had no 'nbart_blue' band). The caller
    (app.py) is expected to hold onto this in st.session_state for the
    rest of the session, so the scene-preview diagnostic overlay can reuse
    it - this replaces an earlier version that cached it to a shared file
    on the server's disk, which risked one user's reference leaking into
    another user's preview if they used the same site name (see this
    module's docstring for the multi-user rationale).

    `min_roi_coverage` (0-1) drops any scene whose tile doesn't cover at
    least that fraction of the drawn ROI polygon - e.g. dates where the ROI
    straddles two tiles collected on different days, or sits right at the
    edge of a swath. Defaults to requiring (near-)full coverage, since a
    partially-covering tile leaves part of the ROI as nodata, which usually
    just produces an indeterminate result rather than a useful one - so
    filtering these out up front both avoids that noise and cuts down the
    number of scenes that need fetching/analysing. Pass 0 to disable this
    filter and keep every scene regardless of coverage.

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
    `nbart_blue` in the product's bands and adds a bit of per-run compute.
    `_percentile`/`_min_obs` control how the reference itself is built -
    see connectivity.build_clear_sky_reference's docstring.
    """
    problems = site.validate()
    if problems:
        raise ValueError("Site layers are not valid: " + "; ".join(problems))

    target_crs = site.target_crs()
    inside = site.inside_line(target_crs)
    outside = site.outside_line(target_crs)
    structures = site.structures_reprojected(target_crs)

    region_mgr = InMemoryRegionManager(site.roi, region_name=site.name)
    stac_mgr = STACDataManager(root_dir=_stac_work_dir())

    records: list[dict] = []
    clear_sky_references: dict[str, Optional[np.ndarray]] = {}

    for product_code in products:
        product_mgr = RSDataProductManager(product_code, product_param_path=products_json_path)

        if progress_cb:
            progress_cb(0, 1, f"Searching {product_code} scenes...")
        items = stac_mgr.get_stac_items(start_date, end_date, product_mgr, region_mgr)
        items = stac_mgr.filter_stac_items_eocloud(items, max_cloud)
        if not items:
            clear_sky_references[product_code] = None
            continue
        if min_roi_coverage > 0:
            items = stac_mgr.filter_stac_items_region_overlap(
                items, min_roi_coverage, region_mgr, target_crs=target_crs
            )
        if not items:
            clear_sky_references[product_code] = None
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
        # the enable_temporal_anomaly docstring above. Kept in memory only
        # and handed back to the caller - never written to disk (see
        # module docstring).
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
            else:
                print(f"Warning: temporal-anomaly detection requested but 'nbart_blue' not in {product_code}'s bands - skipping for this product.")
        clear_sky_references[product_code] = clear_sky_reference

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
                    error=str(e),
                )
                is_error, nodata = True, None

            candidate_key = (is_error, nodata if nodata is not None else float("inf"))
            existing = product_records.get(date)
            if existing is None or candidate_key < existing[0]:
                product_records[date] = (candidate_key, record)

        records.extend(record for _, record in product_records.values())

    return records, clear_sky_references


def fetch_single_scene(
    site: SiteLayers,
    target_date,
    sensor: str,
    products_json_path: str,
):
    """Fetches just the one scene for `target_date` and `sensor`
    ("landsat" or "sentinel2") - used by the app's scene preview to pull a
    raster on demand when a point on the results plot is clicked. Always
    fetches fresh from DEA and returns the result in memory only - nothing
    is written to disk (see module docstring on why: a shared server-side
    cache risks leaking one user's scene into another user's preview). It's
    the caller's job (app.py) to hold onto the single most-recently-fetched
    scene in st.session_state for a fast re-render on trivial reruns (e.g.
    toggling a checkbox), and to drop it as soon as a different scene is
    selected.

    Returns (bands, transform, crs, diagnostics). `diagnostics` is a dict
    with `n_items_found`, `items` (per-item id/datetime/collection/
    cloud_cover/region_overlap_pct, or None if it couldn't be computed) and
    `valid_pixel_frac` (fraction of the clipped ROI that's not oa_fmask
    no-data - a quick way to tell whether a near-blank preview is a real
    coverage gap versus something else going wrong upstream).
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

    region_mgr = InMemoryRegionManager(site.roi, region_name=site.name)
    stac_mgr = STACDataManager(root_dir=_stac_work_dir())
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
    best_bands, best_transform, best_valid_frac = None, None, -1.0
    for cand_idx in matches:
        cand_scene = processed.isel(time=cand_idx)
        cand_bands = {var: np.asarray(cand_scene[var].values) for var in cand_scene.data_vars}
        vf = _valid_pixel_frac(cand_bands) or 0.0
        if vf > best_valid_frac:
            best_bands, best_transform, best_valid_frac = (
                cand_bands, cand_scene.rio.transform(), vf,
            )

    bands, transform = best_bands, best_transform

    diagnostics = {
        "source": "fetched",
        "n_items_found": len(items),
        "n_same_date_candidates": len(matches),
        "items": item_info,
        "valid_pixel_frac": _valid_pixel_frac(bands),
    }
    return bands, transform, rasterio.crs.CRS.from_epsg(target_crs), diagnostics
