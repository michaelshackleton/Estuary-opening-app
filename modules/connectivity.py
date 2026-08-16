"""
Python port of ``Estuary openings multi folders.r``.

This module reproduces the R script's logic for deciding whether an estuary
mouth is open, closed, or indeterminate on a single satellite scene, given:
  - a green + NIR band pair (for NDWI)
  - an oa_fmask band (DEA convention: 0=invalid, 1=valid, 2=cloud,
    3=cloud shadow, 4=snow, 5=water)
  - an "inside" line (river side) and "outside" line (ocean side), rasterised
    onto the same grid as the imagery
  - optional structure polygons (bridges/causeways) rasterised the same way

See the original R script's header comments for the full rationale behind
each rule - the logic here is intentionally a 1:1 translation, not a
re-design, so that results match the R version. Key rules, in brief:

  - oa_fmask == 4 (snow) is treated as misclassified wave/foam and forced to
    water (== 5) before anything else happens.
  - A cell is "unknown" (excluded from the water/land call, not scored as
    either) if oa_fmask is 0 (no-data) or 2 (cloud).
  - Cloud shadow (oa_fmask == 3) is NOT unknown - it is resolved via the
    NDWI test in both the NDWI and fmask variants, since shadow affects a
    band *ratio* like NDWI much less than it affects fmask's own
    absolute-reflectance classification.
  - Connectivity is checked twice per variant: "strict" (unknown cells are
    not water) and "optimistic" (unknown cells might be water). If strict
    connects -> open. If neither connects -> closed. If only optimistic
    connects -> indeterminate (a data gap sits somewhere a path could
    plausibly have run through).
  - Structure polygons (bridges/causeways) force their covered cells to
    count as water and pulls them out of "unknown" status, in both variants.
  - The least-cost "gap" is the minimum number of non-water, non-unknown
    cells that would need to become water to connect the two lines - 0 means
    already connected, 1 is a case worth treating with extra caution (could
    be a coarse-resolution/mixed-pixel artefact).

This module only needs numpy/scipy/rasterio/scikit-image arrays - it has no
Streamlit or STAC/DEA dependency, so it can be unit tested or reused
independently of the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from rasterio.features import rasterize
from scipy import ndimage

try:
    from skimage.graph import MCP
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scikit-image is required for the least-cost gap calculation "
        "(pip install scikit-image)"
    ) from e


FMASK_UNKNOWN_CODES = (0, 2)  # no-data, cloud
FMASK_VALID_CODES = (0, 1, 2, 3, 4, 5)  # every recognised DEA oa_fmask code
FMASK_LABELS = {0: "nodata", 2: "cloud", 3: "cloud_shadow"}

# Default cloud-edge buffer, in pixels, per sensor - see build_cloud_buffer_mask()
# and apply_cloud_buffer(). Chosen to correspond to a roughly similar real-world
# distance across sensors (Sentinel-2 is 10m/px, Landsat 30m/px), based on
# visual inspection of false "closed" calls caused by cloud-edge/haze artefacts
# right along a river channel.
DEFAULT_CLOUD_BUFFER_PX = {"sentinel2": 25, "landsat": 3}

# Default blue-band brightness excess (in raw DN, DEA nbart bands are
# reflectance x10000) above a pixel's own clear-sky reference that counts as
# a "temporal anomaly" - see build_clear_sky_reference() / build_temporal_
# anomaly_mask() below. ~400 DN is roughly a 0.04 reflectance jump - a
# starting point, not a validated figure; this genuinely needs calibrating
# against real scenes (the diagnostic mask overlay in the app is built for
# exactly that) since it trades off catching thin/dappled cloud fmask misses
# against false-flagging pixels that are just naturally variable (e.g.
# turbid water, wet sand at different tide states).
DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN = 400.0


def unknown_from_fmask(fmask_original: np.ndarray) -> np.ndarray:
    """Cells are 'unknown' if fmask flags them no-data/cloud (0/2), OR if
    the value isn't a recognised oa_fmask code at all (0-5). The latter
    catches the nodata sentinel rioxarray writes into cells that fall
    outside the user's drawn ROI polygon but inside its bounding box after
    `.rio.clip()` - without this, those cells would silently be scored as
    'land' instead of being excluded as unknown."""
    return np.isin(fmask_original, FMASK_UNKNOWN_CODES) | ~np.isin(fmask_original, FMASK_VALID_CODES)

IsWaterFn = Callable[[np.ndarray], np.ndarray]


def default_is_water(ndwi: np.ndarray) -> np.ndarray:
    """NDWI > 0 => water (McFeeters 1996). Treat as a starting point, not a
    universal truth - callers can pass their own threshold function."""
    return ndwi > 0


# --------------------------------------------------------------------------
# NDWI + sensor detection
# --------------------------------------------------------------------------

def calc_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (green - nir) / (green + nir)."""
    green = green.astype("float32")
    nir = nir.astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / (green + nir)
    return ndwi


def detect_sensor_and_nir_band(band_names) -> tuple[str, str]:
    """Both Sentinel-2 and Landsat DEA ARD products share 'nbart_green'.
    They differ in the NIR band name: Sentinel-2 = 'nbart_nir_1',
    Landsat = 'nbart_nir'. Returns (sensor, nir_band_name)."""
    has_s2 = "nbart_nir_1" in band_names
    has_ls = "nbart_nir" in band_names
    if has_s2 and has_ls:
        raise ValueError(
            "Both 'nbart_nir_1' and 'nbart_nir' present - specify the NIR "
            "band explicitly rather than relying on auto-detection."
        )
    if has_s2:
        return "sentinel2", "nbart_nir_1"
    if has_ls:
        return "landsat", "nbart_nir"
    raise ValueError(
        "Could not find a recognised NIR band ('nbart_nir_1' for "
        f"Sentinel-2 or 'nbart_nir' for Landsat). Available bands: "
        f"{sorted(band_names)}"
    )


# --------------------------------------------------------------------------
# fmask handling
# --------------------------------------------------------------------------

def build_updated_fmask(fmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (original, updated) fmask arrays. 'updated' reclassifies
    snow/ice (4) as water (5) - in DEA coastal scenes, class 4 is wave
    crests/foam being misclassified as snow, not genuine snow/ice."""
    original = fmask
    updated = np.where(original == 4, 5, original)
    return original, updated


def build_ndwi_water_mask(
    fmask_original: np.ndarray, ndwi: np.ndarray, is_water: IsWaterFn = default_is_water
) -> tuple[np.ndarray, np.ndarray]:
    """NDWI-based water mask. NDWI is tested on clear (1), cloud shadow (3)
    and already-water (5) cells; former-snow/wave cells (4) are forced to
    water regardless of NDWI. fmask 0/2 (no-data/cloud) are unknown."""
    ndwi_water = is_water(ndwi) & np.isin(fmask_original, [1, 3, 5])
    forced_water = fmask_original == 4
    water_strict = ndwi_water | forced_water
    unknown_mask = unknown_from_fmask(fmask_original)
    return water_strict, unknown_mask


def build_fmask_water_mask(
    fmask_original: np.ndarray,
    fmask_updated: np.ndarray,
    ndwi: np.ndarray,
    is_water: IsWaterFn = default_is_water,
) -> tuple[np.ndarray, np.ndarray]:
    """fmask-based water mask: cells fmask itself calls water (==5, after
    the snow->water fix), plus cloud-shadow cells (3) resolved via NDWI
    rather than left unknown (shadow cells failing the NDWI test are scored
    as land here, since NDWI - not fmask - is doing the classifying)."""
    fmask_water = fmask_updated == 5
    shadow_ndwi_water = (fmask_original == 3) & is_water(ndwi)
    water_strict = fmask_water | shadow_ndwi_water
    unknown_mask = unknown_from_fmask(fmask_original)
    return water_strict, unknown_mask


# --------------------------------------------------------------------------
# Cloud-edge buffer
# --------------------------------------------------------------------------
#
# fmask's own cloud (2) / cloud shadow (3) classification only ever covers
# the core of a cloud - the ring of pixels immediately around a cloud's edge
# (haze, thin cirrus, mixed cloud/land pixels at coarser resolution) often
# gets called "clear" (1) by fmask, but is still radiometrically
# contaminated enough to throw NDWI's land/water call off. Right along a
# river channel, this produces thin false "dry" slivers exactly where a
# cloud edge crosses the water - which can wrongly break an otherwise-open
# connection between the inside and outside lines and call the scene
# "closed" when it's really just occluded. Dilating the cloud mask by a few
# pixels and treating that ring as "unknown" (rather than trusting a "land"/
# "dry" call there) turns that false "closed" into a correctly cautious
# "indeterminate" instead. Cells the buffer overlaps that NDWI/fmask
# already called water are a different matter entirely - there's no
# "false dry" ambiguity to guard against for those, so the buffer leaves
# them alone (see apply_cloud_buffer below) rather than needlessly
# demoting an already-positive water call to "unknown".

def build_cloud_buffer_mask(fmask_original: np.ndarray, buffer_px: int) -> np.ndarray:
    """Boolean mask of cells within `buffer_px` pixels (Chebyshev distance,
    i.e. a square structuring element - matches how ndimage.binary_dilation
    with a square footprint grows a region) of a cloud cell (oa_fmask==2).
    Returns an all-False mask if buffer_px <= 0 or there's no cloud in the
    scene at all."""
    cloud = fmask_original == 2
    if buffer_px <= 0 or not cloud.any():
        return np.zeros_like(cloud, dtype=bool)
    structure = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), dtype=bool)
    return ndimage.binary_dilation(cloud, structure=structure)


def apply_cloud_buffer(
    water_strict: np.ndarray, unknown_mask: np.ndarray, cloud_buffer_mask: Optional[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Cells within the cloud buffer zone that are NOT already water (by
    NDWI or fmask - i.e. `water_strict` is already False there) are forced
    to 'unknown'. Cells the buffer overlaps that ARE already water are left
    completely alone - the buffer exists only to catch false "dry"
    cloud-edge artefacts (haze making real water/land wrongly look dry), not
    to doubt an already-positive water call, so it should never strip a
    water_strict claim. (An earlier version did strip water_strict here
    unconditionally, which wrongly turned real open-water pixels right next
    to a cloud into "unknown" purely for being near one.) Run this *before*
    apply_structure_mask, not after, so that a known structure (bridge/
    causeway) still overrides a cloud buffer sitting on top of it - a
    structure's water status is known regardless of nearby cloud."""
    if cloud_buffer_mask is None or not cloud_buffer_mask.any():
        return water_strict, unknown_mask
    downgrade = cloud_buffer_mask & ~water_strict
    return water_strict, unknown_mask | downgrade


# --------------------------------------------------------------------------
# Temporal cloud/haze anomaly detection
# --------------------------------------------------------------------------
#
# The cloud-edge buffer above only ever grows outward from cells fmask
# itself already flagged as cloud (oa_fmask==2) - it does nothing for cloud
# fmask never detected in the first place, which happens routinely for thin,
# dappled or broken cloud: fmask's own algorithm only catches a scattered
# fraction of it, leaving most of the contaminated area reading as ordinary
# "clear" land/water with no cloud flag anywhere nearby to dilate from.
#
# This is a different, complementary check: instead of trusting fmask's
# classification at all, it asks "does this pixel look anomalously bright,
# compared to how this *exact* pixel has looked on other, clearer dates at
# this site?" Cloud and haze brighten a pixel (especially in the blue band,
# which is most sensitive to atmospheric scattering) well before there's
# enough of it for fmask's own cloud algorithm to trigger, so this can catch
# contamination fmask misses entirely - at the cost of needing a decent
# amount of clear-observation history at each pixel to build a reliable
# per-pixel reference from, and a threshold that will need calibrating
# against real scenes (see the app's diagnostic mask overlay).
#
# Whitecaps/wave foam are a specific false-positive risk worth calling out:
# they're bright enough in the blue band to look exactly like haze, but
# they're just turbulent water, not contamination. fmask already has a
# convention for this - it misclassifies wave crests as "snow" (oa_fmask==4)
# - and the water-mask builders below already force snow to water for
# exactly that reason. apply_temporal_anomaly() relies on that: a cell
# that's already water_strict (which snow-forced-water is, well before this
# check ever runs) is never downgraded by an anomaly flag, regardless of how
# bright it reads - so waves don't need any special-casing here.

def build_clear_sky_reference(
    blue_stack: np.ndarray, fmask_stack: np.ndarray, percentile: float = 10.0, min_clear_obs: int = 3
) -> np.ndarray:
    """`blue_stack`/`fmask_stack` are (time, y, x) arrays - the blue band
    and oa_fmask for every scene in a site's time series, already on a
    common grid (same transform/shape for every time slice - true of the
    per-product `processed` dataset in fetch.run_site_analysis(), which is
    exactly where this is meant to be called from).

    Returns a 2D 'clear-sky reference' - the low percentile of blue-band
    values across every observation fmask itself called clear land (1) or
    water (5) at that pixel, i.e. roughly the least atmosphere/haze-
    contaminated value seen there historically. A *low* percentile (not the
    mean/median) is deliberate: cloud/haze only ever brightens a pixel, so
    the lower tail of its history is far less likely to already be
    contaminated itself than a central-tendency summary would be; a low
    percentile rather than the bare minimum is a small safety margin
    against a single unusually-dark outlier observation.

    NaN wherever fewer than `min_clear_obs` clear/water observations were
    available at that pixel - such a pixel simply gets no reference (and
    therefore no temporal-anomaly check at all - see
    build_temporal_anomaly_mask()) rather than a fabricated, unreliable one.
    """
    clear_or_water = np.isin(fmask_stack, [1, 5])
    n_obs = clear_or_water.sum(axis=0)

    blue = blue_stack.astype(np.float32)
    masked = np.where(clear_or_water, blue, np.nan)
    with np.errstate(all="ignore"):  # all-NaN slices (pixel never clear) are expected, not a bug
        reference = np.nanpercentile(masked, percentile, axis=0)
    reference = np.where(n_obs >= min_clear_obs, reference, np.nan)
    return reference.astype(np.float32)


def build_temporal_anomaly_mask(
    blue: np.ndarray,
    reference: Optional[np.ndarray],
    threshold_dn: float = DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN,
) -> np.ndarray:
    """Flags pixels whose blue-band reflectance in this scene sits more
    than `threshold_dn` above their own clear-sky reference (see
    build_clear_sky_reference above) - independent of what fmask says about
    them. Returns an all-False mask if no reference is available at all
    (e.g. `reference` is None, meaning temporal-anomaly checking wasn't
    enabled for this run)."""
    if reference is None:
        return np.zeros_like(blue, dtype=bool)
    has_reference = ~np.isnan(reference)
    excess = blue.astype(np.float32) - reference
    return has_reference & (excess > threshold_dn)


def apply_temporal_anomaly(
    water_strict: np.ndarray, unknown_mask: np.ndarray, anomaly_mask: Optional[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Same pattern as apply_cloud_buffer() above, and for the same reason:
    cells flagged as a temporal brightness anomaly that are NOT already
    water (by NDWI or fmask) are forced to 'unknown'; cells that ARE already
    water are left alone, since an anomalously-bright pixel that's already
    confirmed water by another method isn't ambiguous - it's just bright
    water. This is also what makes wave/foam cells safe: fmask flags wave
    crests as "snow" (oa_fmask==4), which build_updated_fmask/
    build_ndwi_water_mask/build_fmask_water_mask already force into
    water_strict *before* this function ever runs, specifically because
    that misclassification is common right here. Foam is bright enough to
    trip the brightness-anomaly check on its own, but since it's already
    water_strict by the time this runs, it's never downgraded - so waves no
    longer get wrongly flagged as cloud/haze contamination. (An earlier
    version stripped water_strict unconditionally here too, which is what
    caused that.) Run before apply_structure_mask, not after, so a known
    structure (bridge/causeway) still overrides an anomaly flag sitting on
    top of it."""
    if anomaly_mask is None or not anomaly_mask.any():
        return water_strict, unknown_mask
    downgrade = anomaly_mask & ~water_strict
    return water_strict, unknown_mask | downgrade


# --------------------------------------------------------------------------
# Structures (bridges/causeways)
# --------------------------------------------------------------------------

def build_structure_mask(shape, transform, structures_gdf) -> Optional[np.ndarray]:
    """Rasterises structure polygons onto the scene's grid. Returns None if
    no structures are supplied (so callers can skip the override with a
    simple `is None` check), matching the R script's `resolve_structures`
    / `build_structure_mask` pairing."""
    if structures_gdf is None or len(structures_gdf) == 0:
        return None
    shapes = [(geom, 1) for geom in structures_gdf.geometry]
    arr = rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8"
    )
    return arr.astype(bool)


def apply_structure_mask(
    water_strict: np.ndarray, unknown_mask: np.ndarray, structure_mask: Optional[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Cells under a structure are forced to water AND removed from
    'unknown', even if they were cloud/shadow/no-data - a structure cell is
    never actually ambiguous, we know it spans water."""
    if structure_mask is None:
        return water_strict, unknown_mask
    return water_strict | structure_mask, unknown_mask & ~structure_mask


# --------------------------------------------------------------------------
# Line rasterisation
# --------------------------------------------------------------------------

def rasterize_line(line_gdf, shape, transform, all_touched: bool = True) -> np.ndarray:
    shapes = [(geom, 1) for geom in line_gdf.geometry]
    arr = rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, all_touched=all_touched, dtype="uint8"
    )
    return arr.astype(bool)


# --------------------------------------------------------------------------
# Connectivity
# --------------------------------------------------------------------------

@dataclass
class ConnectivityResult:
    connected: bool
    shared_patches: set
    inside_patches: set
    outside_patches: set
    patch_raster: np.ndarray


def find_connectivity(
    water_bool: np.ndarray, inside_mask: np.ndarray, outside_mask: np.ndarray, connectivity: int = 8
) -> ConnectivityResult:
    """Label connected water patches (8-connectivity by default, i.e.
    diagonal neighbours count as connected - matches `directions = 8` in the
    R script) and test whether the inside/outside lines touch the same
    patch."""
    structure = np.ones((3, 3), dtype=int) if connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    labeled, _ = ndimage.label(water_bool, structure=structure)

    inside_patches = set(np.unique(labeled[inside_mask & (labeled > 0)]).tolist())
    outside_patches = set(np.unique(labeled[outside_mask & (labeled > 0)]).tolist())
    shared = inside_patches & outside_patches

    return ConnectivityResult(
        connected=len(shared) > 0,
        shared_patches=shared,
        inside_patches=inside_patches,
        outside_patches=outside_patches,
        patch_raster=labeled,
    )


@dataclass
class IndeterminateResult:
    status: str  # "TRUE" | "FALSE" | "indeterminate"
    strict: ConnectivityResult
    optimistic: Optional[ConnectivityResult]
    indeterminate_reason: Optional[str]


def check_connectivity_indeterminate(
    water_strict: np.ndarray,
    unknown_mask: np.ndarray,
    fmask_original: np.ndarray,
    inside_mask: np.ndarray,
    outside_mask: np.ndarray,
    connectivity: int = 8,
    cloud_buffer_mask: Optional[np.ndarray] = None,
    temporal_anomaly_mask: Optional[np.ndarray] = None,
) -> IndeterminateResult:
    strict = find_connectivity(water_strict, inside_mask, outside_mask, connectivity)
    if strict.connected:
        return IndeterminateResult("TRUE", strict, None, None)

    optimistic_water = water_strict | unknown_mask
    optimistic = find_connectivity(optimistic_water, inside_mask, outside_mask, connectivity)
    if not optimistic.connected:
        return IndeterminateResult("FALSE", strict, optimistic, None)

    # indeterminate - work out which unknown class(es) sit on the patch
    # that bridges the two lines in the optimistic pass
    bridge_mask = np.isin(optimistic.patch_raster, list(optimistic.shared_patches)) & unknown_mask
    bridge_vals = fmask_original[bridge_mask]
    codes_present = sorted(set(int(v) for v in np.unique(bridge_vals)))
    labels = [FMASK_LABELS.get(c, "outside_roi") for c in codes_present if c not in (1, 5)]
    # A bridging cell can be "unknown" purely because it fell in the cloud
    # buffer zone while itself reading as clear (1) or water (5) fmask - the
    # loop above skips those codes entirely, so without this check a
    # buffer-only bridge would silently report no reason at all.
    if cloud_buffer_mask is not None and np.any(bridge_mask & cloud_buffer_mask) and "cloud_buffer" not in labels:
        labels.append("cloud_buffer")
    # Same reasoning as the cloud_buffer check above, for the temporal
    # brightness-anomaly flag - a bridging cell can be "unknown" purely
    # because it was flagged anomalously bright relative to its own history,
    # while itself reading as clear (1) or water (5) fmask.
    if temporal_anomaly_mask is not None and np.any(bridge_mask & temporal_anomaly_mask) and "temporal_anomaly" not in labels:
        labels.append("temporal_anomaly")
    reason = "+".join(labels) if labels else None

    return IndeterminateResult("indeterminate", strict, optimistic, reason)


def connectivity_gap_cells(
    water_strict: np.ndarray,
    unknown_mask: np.ndarray,
    inside_mask: np.ndarray,
    outside_mask: np.ndarray,
) -> Optional[float]:
    """Least-cost path: water cells cost 0 to enter, every other valid cell
    costs 1, unknown cells are impassable. Returns the minimum number of
    non-water cells that would need to become water to connect the lines.
    0 = already connected. 1 is worth flagging (possible mixed-pixel
    artefact). None = every route is blocked by unknown cells, or a line
    doesn't touch any valid cell at all - matches the R script's NA case.

    Uses skimage.graph.MCP (not MCP_Geometric) so that diagonal steps cost
    the same as orthogonal steps - the cost model here is "how many cells
    were entered", not physical distance, matching the R igraph
    implementation exactly.
    """
    cost = np.where(water_strict, 0.0, 1.0).astype("float64")
    cost = np.where(unknown_mask, np.inf, cost)

    valid = np.isfinite(cost)
    inside_cells = np.argwhere(inside_mask & valid)
    outside_cells = np.argwhere(outside_mask & valid)

    if len(inside_cells) == 0 or len(outside_cells) == 0:
        return None

    mcp = MCP(cost, fully_connected=True)
    starts = [tuple(c) for c in inside_cells]
    ends = [tuple(c) for c in outside_cells]
    cost_array, _ = mcp.find_costs(starts, ends)

    best = min(cost_array[r, c] for r, c in ends)
    return None if not np.isfinite(best) else float(best)


# --------------------------------------------------------------------------
# Combine NDWI + fmask into a single open/closed/indeterminate call, plus
# per-scene diagnostics
# --------------------------------------------------------------------------

def combine_status(status_ndwi: str, status_fmask: str) -> str:
    """A path via *either* NDWI or oa_fmask==5 is enough to call the
    estuary open (per the app spec: 'find whether a NDWI or oa_fmask == 5
    path can be built ... and if so the mouth will be defined as open').
    Closed only if neither variant can connect and neither is blocked by
    unknown cells; indeterminate if either variant is indeterminate and
    neither is a definite TRUE."""
    if status_ndwi == "TRUE" or status_fmask == "TRUE":
        return "open"
    if status_ndwi == "FALSE" and status_fmask == "FALSE":
        return "closed"
    return "indeterminate"


@dataclass
class SceneResult:
    sensor: str
    status: str  # "open" | "closed" | "indeterminate"
    status_ndwi: str
    status_fmask: str
    gap_ndwi: Optional[float]
    gap_fmask: Optional[float]
    reason_ndwi: Optional[str]
    reason_fmask: Optional[str]
    n_nodata_cells: int
    pct_cloud: float
    pct_cloud_shadow: float
    patch_raster_ndwi: np.ndarray = field(repr=False)
    ndwi: np.ndarray = field(repr=False)
    cloud_buffer_px: int = 0
    pct_temporal_anomaly: float = 0.0  # % of scene flagged as a temporal brightness anomaly
    temporal_reference_coverage_pct: float = 0.0  # % of scene that had a usable clear-sky reference at all
    # Full check objects (each has .strict and .optimistic ConnectivityResult,
    # with .patch_raster / .shared_patches / .inside_patches / .outside_patches)
    # kept so callers can inspect *why* a status was reached - e.g. to check
    # whether the optimistic (cloud/no-data-as-water) pass genuinely found no
    # connecting patch, rather than just trusting the summary status string.
    ndwi_check: "IndeterminateResult" = field(repr=False, default=None)
    fmask_check: "IndeterminateResult" = field(repr=False, default=None)


def process_scene(
    bands: dict,
    transform,
    inside_gdf,
    outside_gdf,
    structures_gdf=None,
    is_water: IsWaterFn = default_is_water,
    connectivity: int = 8,
    cloud_buffer_px: Optional[int | dict] = None,
    clear_sky_reference: Optional[np.ndarray] = None,
    temporal_anomaly_threshold: float = DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN,
) -> SceneResult:
    """Process a single scene end-to-end. `bands` is a dict of 2D numpy
    arrays keyed by DEA band name, must include 'nbart_green', one of
    'nbart_nir'/'nbart_nir_1', and 'oa_fmask'. `transform` is the scene's
    affine transform (e.g. from rioxarray's `.rio.transform()`). inside/
    outside/structures are geopandas GeoDataFrames already reprojected to
    match the scene's CRS.

    `cloud_buffer_px` controls the cloud-edge buffer (see the "Cloud-edge
    buffer" section above) - pass an int to use that value for every scene
    regardless of sensor, a dict like {"sentinel2": 25, "landsat": 3} to
    vary it by sensor (the detected sensor's entry is used, falling back to
    DEFAULT_CLOUD_BUFFER_PX if that sensor isn't a key), or leave as None to
    use DEFAULT_CLOUD_BUFFER_PX outright. Pass 0 to disable buffering
    entirely and reproduce the original (pre-buffer) behaviour.

    `clear_sky_reference` is the per-pixel reference built by
    build_clear_sky_reference() from this scene's full site time series
    (see the "Temporal cloud/haze anomaly detection" section above) -
    leave as None (the default) to skip temporal-anomaly checking entirely,
    which is exactly what happens if it wasn't enabled/available for this
    run. Requires 'nbart_blue' in `bands` to have any effect.
    `temporal_anomaly_threshold` is the blue-band DN excess above the
    reference that counts as anomalous - see
    DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN's docstring for caveats."""
    sensor, nir_band = detect_sensor_and_nir_band(bands.keys())
    green = bands["nbart_green"]
    nir = bands[nir_band]
    fmask_original_raw = bands["oa_fmask"]

    ndwi = calc_ndwi(green, nir)
    fmask_original, fmask_updated = build_updated_fmask(fmask_original_raw)

    shape = fmask_original.shape
    structure_mask = build_structure_mask(shape, transform, structures_gdf)

    if cloud_buffer_px is None:
        buffer_px = DEFAULT_CLOUD_BUFFER_PX.get(sensor, 0)
    elif isinstance(cloud_buffer_px, dict):
        buffer_px = cloud_buffer_px.get(sensor, DEFAULT_CLOUD_BUFFER_PX.get(sensor, 0))
    else:
        buffer_px = int(cloud_buffer_px)
    cloud_buffer_mask = build_cloud_buffer_mask(fmask_original, buffer_px)

    temporal_anomaly_mask = None
    if clear_sky_reference is not None and "nbart_blue" in bands:
        temporal_anomaly_mask = build_temporal_anomaly_mask(
            bands["nbart_blue"], clear_sky_reference, temporal_anomaly_threshold
        )

    ndwi_water_strict, ndwi_unknown = apply_structure_mask(
        *apply_temporal_anomaly(
            *apply_cloud_buffer(*build_ndwi_water_mask(fmask_original, ndwi, is_water), cloud_buffer_mask),
            temporal_anomaly_mask,
        ),
        structure_mask,
    )
    fmask_water_strict, fmask_unknown = apply_structure_mask(
        *apply_temporal_anomaly(
            *apply_cloud_buffer(*build_fmask_water_mask(fmask_original, fmask_updated, ndwi, is_water), cloud_buffer_mask),
            temporal_anomaly_mask,
        ),
        structure_mask,
    )

    inside_mask = rasterize_line(inside_gdf, shape, transform)
    outside_mask = rasterize_line(outside_gdf, shape, transform)

    ndwi_check = check_connectivity_indeterminate(
        ndwi_water_strict, ndwi_unknown, fmask_original, inside_mask, outside_mask,
        connectivity, cloud_buffer_mask, temporal_anomaly_mask,
    )
    fmask_check = check_connectivity_indeterminate(
        fmask_water_strict, fmask_unknown, fmask_original, inside_mask, outside_mask,
        connectivity, cloud_buffer_mask, temporal_anomaly_mask,
    )

    gap_ndwi = connectivity_gap_cells(ndwi_water_strict, ndwi_unknown, inside_mask, outside_mask)
    gap_fmask = connectivity_gap_cells(fmask_water_strict, fmask_unknown, inside_mask, outside_mask)

    n_cells = fmask_original.size
    n_nodata = int(np.sum(fmask_original == 0))
    pct_cloud = 100.0 * float(np.sum(fmask_original == 2)) / n_cells
    pct_cloud_shadow = 100.0 * float(np.sum(fmask_original == 3)) / n_cells
    pct_temporal_anomaly = (
        100.0 * float(temporal_anomaly_mask.sum()) / n_cells if temporal_anomaly_mask is not None else 0.0
    )
    temporal_reference_coverage_pct = (
        100.0 * float(np.sum(~np.isnan(clear_sky_reference))) / n_cells
        if clear_sky_reference is not None else 0.0
    )

    return SceneResult(
        sensor=sensor,
        status=combine_status(ndwi_check.status, fmask_check.status),
        status_ndwi=ndwi_check.status,
        status_fmask=fmask_check.status,
        gap_ndwi=gap_ndwi,
        gap_fmask=gap_fmask,
        reason_ndwi=ndwi_check.indeterminate_reason,
        reason_fmask=fmask_check.indeterminate_reason,
        n_nodata_cells=n_nodata,
        pct_cloud=pct_cloud,
        pct_cloud_shadow=pct_cloud_shadow,
        patch_raster_ndwi=ndwi_check.strict.patch_raster,
        ndwi=ndwi,
        cloud_buffer_px=buffer_px,
        pct_temporal_anomaly=pct_temporal_anomaly,
        temporal_reference_coverage_pct=temporal_reference_coverage_pct,
        ndwi_check=ndwi_check,
        fmask_check=fmask_check,
    )
