"""
Estuary Mouth Monitor - Streamlit app (cloud-hosting build).

Draw a region of interest around an estuary, an inside/outside line pair,
and (optionally) polygons around structures like bridges, then fetch every
available Landsat/Sentinel-2 scene from Digital Earth Australia for a date
range and cloud-cover threshold, and classify the estuary mouth as
open/closed/indeterminate on each scene. See modules/connectivity.py for the
classification logic (a Python port of "Estuary openings multi folders.r")
and modules/fetch.py for the DEA data retrieval.

This is a fork of the desktop "Claude_script" app, adapted to run on
Streamlit Community Cloud (streamlit.io) - see this folder's README.md for
exactly what's different and why. The short version: site-layer save/load
now goes through a downloaded/uploaded JSON file (modules/region.py's
to_json_bytes/from_json_bytes) instead of native OS folder dialogs and
Shapefiles, since a server has no access to the user's own filesystem and
there's no display to pop a dialog on. Everything else - the map, the
connectivity algorithm, the DEA fetch pipeline - is unchanged.

Run with:  streamlit run app.py   (see README.md for local setup, or deploy
directly to Streamlit Community Cloud by pointing it at this file)
"""

from __future__ import annotations

import os
import sys
from datetime import date

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from rasterio.transform import rowcol
from streamlit_folium import st_folium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules import aggregate, connectivity, fetch  # noqa: E402
from modules.region import SiteLayers, geojson_features_to_gdf, name_from_centroid  # noqa: E402
import io
import zipfile

PRODUCTS_JSON = os.path.join(BASE_DIR, "config", "products.json")
MURRAY_MOUTH_IMAGE = os.path.join(BASE_DIR, "Murray_mouth.jpg")

AUSTRALIA_CENTER = (-25.27, 133.775)
LANDSAT5_START = date(1985, 3, 1)

# Pastel colour palette used across the maps and plots.
PASTEL = dict(
    roi="#7FB3D5",           # light blue
    inside="#6FA8DC",        # pastel blue
    outside="#F4A6B7",       # pastel rose (paired with inside for contrast)
    structures="#FFCC99",    # pastel peach
    sandbars="#E0C68C",      # pastel sandy tan - distinct from structures' peach
    landsat="#A9CCE3",       # pastel blue
    sentinel="#F4A6B7",      # pastel rose
    nodata="#D6D6D6",        # light grey
    cloud="#CBB4E0",         # pastel lavender
    cloud_shadow="#AEE3F0",  # pastel sky blue
    outside_roi="#FCE8A8",   # pastel yellow
    temporal_anomaly="#F2B279",  # pastel orange - flagged by the temporal clear-sky comparison, not fmask
    cloud_buffer="#D9B872",  # dusty tan/gold - inside a cloud's dilated buffer zone, not fmask/NDWI water
    protected_water="#00BFA5",  # teal - NDWI/fmask already calls this water, so a nearby cloud/anomaly flag doesn't exclude it
    land="#E8DCC3",          # pastel sand - fmask valid/land
    water="#A0C4E8",         # pastel blue - fmask water (== 5)
    snow_water="#B5E8D5",    # pastel mint - fmask snow (4, treated as water)
)


def discrete_colorscale(colors: list[str]) -> list:
    """Builds a Plotly stepped colorscale with `len(colors)` equal-width
    bins, so an integer-coded z array renders as hard-edged categories
    rather than a smooth gradient between them. Pair with zmin/zmax set so
    each integer code sits in the middle of its own bin (e.g. codes 0..n-1
    with zmin=-0.5, zmax=n-0.5; or codes 1..n with zmin=0.5, zmax=n+0.5)."""
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    return scale

st.set_page_config(page_title="Estuary Mouth Monitor", layout="wide")


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def init_state():
    defaults = dict(
        roi_gdf=None,
        lines_gdf=None,
        structures_gdf=None,
        sandbars_gdf=None,
        site_name="new_site",
        results_df=None,
        raw_records=None,
        run_message="",
        selected_point=None,
        draw_map_v=0,
        # Where the draw tab's map stays centred once the in-progress site's
        # own roi_gdf is cleared (e.g. right after queueing) - see
        # update_last_site_center().
        last_site_center=None,
        structures_decided=False,
        sandbars_decided=False,
        enable_temporal_anomaly=False,
        temporal_anomaly_threshold=connectivity.DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN,
        temporal_anomaly_percentile=10.0,
        temporal_anomaly_min_obs=3,
        cloud_buffer_s2=connectivity.DEFAULT_CLOUD_BUFFER_PX["sentinel2"],
        cloud_buffer_ls=connectivity.DEFAULT_CLOUD_BUFFER_PX["landsat"],
        # Per-product clear-sky brightness reference built by the most
        # recent "Run analysis" (see fetch.run_site_analysis) - kept only
        # in this session's memory, never written to the server's disk,
        # since it's specific to this user's own ROI/run. {} until a run
        # with temporal-anomaly detection enabled has completed.
        clear_sky_references={},
        # The single scene currently shown in "Scene preview" below, kept
        # only in this session's memory (see modules/fetch.py's docstring
        # for why nothing is cached to the server's disk). Overwritten -
        # and the old scene's arrays dropped - as soon as a different
        # point is selected; never persists past this browser session.
        preview_scene=None,
        # Sites queued up for a batch run - a list of SiteLayers objects,
        # built up in-memory this session (see "Add this site to the batch
        # queue" in the sidebar) or bulk-uploaded from several site JSON
        # files at once. Never written to the server's disk - just like
        # everything else here, it's gone when the browser session ends,
        # so download the combined/per-site CSVs (see tab_batch) if you
        # want to keep the results.
        site_queue=[],
        batch_combined_df=None,
        batch_per_site_dfs=None,
        batch_per_site_errors=None,
        uploaded_results_df=None,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# --------------------------------------------------------------------------
# Map helpers
# --------------------------------------------------------------------------

def satellite_map(center=AUSTRALIA_CENTER, zoom=5):
    m = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite",
        overlay=False,
        control=False,
    ).add_to(m)
    return m


def add_context_layers(m, roi=None, lines=None, structures=None, sandbars=None, alpha=1.0):
    """Adds already-confirmed layers to a map as static (non-editable)
    context so the user can see what they've already drawn while drawing
    the next layer. `alpha` scales both fill and stroke opacity - used to
    render previously-queued sites faded (0.3) alongside whichever site is
    currently being drawn (1.0, the default), so earlier sites stay visible
    as spatial context on the shared map without being mistaken for the
    active drawing."""
    if roi is not None and len(roi) > 0:
        folium.GeoJson(
            roi, name="ROI",
            style_function=lambda f: {"color": PASTEL["roi"], "weight": 3, "opacity": alpha, "fillOpacity": 0.08 * alpha},
        ).add_to(m)
    if lines is not None and len(lines) > 0:
        for _, row in lines.iterrows():
            color = PASTEL["inside"] if row.get("position") == "inside" else PASTEL["outside"]
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda f, c=color: {"color": c, "weight": 5, "opacity": alpha},
            ).add_to(m)
    if structures is not None and len(structures) > 0:
        folium.GeoJson(
            structures, name="Structures",
            style_function=lambda f: {"color": PASTEL["structures"], "weight": 2, "opacity": alpha, "fillOpacity": 0.35 * alpha},
        ).add_to(m)
    if sandbars is not None and len(sandbars) > 0:
        folium.GeoJson(
            sandbars, name="Sandbars",
            style_function=lambda f: {"color": PASTEL["sandbars"], "weight": 2, "opacity": alpha, "fillOpacity": 0.35 * alpha, "dashArray": "4"},
        ).add_to(m)
    return m


def map_center_from_roi():
    if st.session_state.roi_gdf is not None and len(st.session_state.roi_gdf) > 0:
        c = st.session_state.roi_gdf.geometry.iloc[0].centroid
        return (c.y, c.x)
    return AUSTRALIA_CENTER


def update_last_site_center(roi_gdf):
    """Remembers where the map should stay centred once the in-progress
    site's own ROI is cleared (e.g. right after queueing) - without this,
    the draw tab's map would snap back out to the whole-of-Australia view
    every time 'Add to batch queue' resets the current site, instead of
    staying put on the estuary just drawn."""
    if roi_gdf is not None and len(roi_gdf) > 0:
        c = roi_gdf.geometry.iloc[0].centroid
        st.session_state.last_site_center = (c.y, c.x)


# --------------------------------------------------------------------------
# Draw-tab step state machine - derived entirely from what's already been
# confirmed (rather than a separately-tracked "current step" variable), so
# redrawing an earlier layer (e.g. clicking a confirmed ROI button to redraw
# it) automatically rewinds to the right step with no extra bookkeeping.
# --------------------------------------------------------------------------

DRAW_STEPS = ["roi", "inside", "outside", "structures", "sandbars"]
DRAW_STEP_LABELS = {
    "roi": "Region of interest",
    "inside": "Inside line",
    "outside": "Outside line",
    "structures": "Structures",
    "sandbars": "Sandbars",
}


def has_inside_line() -> bool:
    lg = st.session_state.lines_gdf
    return lg is not None and len(lg) > 0 and (lg["position"] == "inside").any()


def has_outside_line() -> bool:
    lg = st.session_state.lines_gdf
    return lg is not None and len(lg) > 0 and (lg["position"] == "outside").any()


def draw_step_confirmed() -> dict:
    return {
        "roi": st.session_state.roi_gdf is not None,
        "inside": has_inside_line(),
        "outside": has_outside_line(),
        "structures": st.session_state.structures_decided,
        "sandbars": st.session_state.sandbars_decided,
    }


def current_draw_step() -> str:
    """The first not-yet-confirmed step, in order - or 'ready' once all
    five are done and the site can be queued for batch."""
    confirmed = draw_step_confirmed()
    for s in DRAW_STEPS:
        if not confirmed[s]:
            return s
    return "ready"


def confirm_draw_step(step_name: str, candidates: list[dict]):
    """Commits whatever's been drawn for `step_name` into session state and
    advances the wizard. `candidates` is the list of GeoJSON features (from
    the map's all_drawings) matching this step's expected geometry type.
    Structures/sandbars can be confirmed with an empty `candidates` list
    (meaning "none at this site") - ROI/lines cannot, since they're
    mandatory for the analysis to run at all."""
    if step_name == "roi":
        if not candidates:
            st.warning("Draw a polygon (or rectangle) around the estuary mouth first.")
            return
        gdf = geojson_features_to_gdf([candidates[-1]])
        st.session_state.roi_gdf = gdf
        # Auto-name the site from this ROI's centroid, so sites don't need a
        # manually-typed unique name - especially useful once you're drawing
        # several sites in one session for a batch run.
        st.session_state.site_name = name_from_centroid(gdf)
        update_last_site_center(gdf)
    elif step_name in ("inside", "outside"):
        if not candidates:
            st.warning(f"Draw the {step_name} line first.")
            return
        new_row = geojson_features_to_gdf([candidates[-1]])
        new_row["position"] = step_name
        other_pos = "outside" if step_name == "inside" else "inside"
        existing = st.session_state.lines_gdf
        other_part = existing[existing["position"] == other_pos] if existing is not None and len(existing) > 0 else None
        parts = [p for p in (new_row, other_part) if p is not None and len(p) > 0]
        st.session_state.lines_gdf = pd.concat(parts, ignore_index=True)
    elif step_name == "structures":
        st.session_state.structures_gdf = geojson_features_to_gdf(candidates) if candidates else None
        st.session_state.structures_decided = True
    elif step_name == "sandbars":
        st.session_state.sandbars_gdf = geojson_features_to_gdf(candidates) if candidates else None
        st.session_state.sandbars_decided = True
    st.session_state.draw_map_v += 1
    st.rerun()


def redraw_draw_step(step_name: str):
    """Reopens an already-confirmed step for editing - only touches that
    step's own layer, leaving other confirmed steps alone (e.g. redrawing
    the ROI doesn't clear already-confirmed lines/structures/sandbars)."""
    if step_name == "roi":
        st.session_state.roi_gdf = None
    elif step_name in ("inside", "outside"):
        existing = st.session_state.lines_gdf
        if existing is not None and len(existing) > 0:
            remaining = existing[existing["position"] != step_name]
            st.session_state.lines_gdf = remaining if len(remaining) > 0 else None
    elif step_name == "structures":
        st.session_state.structures_gdf = None
        st.session_state.structures_decided = False
    elif step_name == "sandbars":
        st.session_state.sandbars_gdf = None
        st.session_state.sandbars_decided = False
    st.session_state.draw_map_v += 1
    st.rerun()


def reset_current_site(new_map_v: bool = True):
    """Clears the in-progress site's drawn layers so a new (blank) site is
    ready to draw - used both by the sidebar's explicit 'Start a new (blank)
    site' button and automatically after 'Add to batch queue', since
    queueing a site should immediately clear the way for the next one
    rather than requiring a separate manual reset click."""
    st.session_state.roi_gdf = None
    st.session_state.lines_gdf = None
    st.session_state.structures_gdf = None
    st.session_state.sandbars_gdf = None
    st.session_state.structures_decided = False
    st.session_state.sandbars_decided = False
    st.session_state.results_df = None
    st.session_state.site_name = "new_site"
    st.session_state.clear_sky_references = {}
    st.session_state.preview_scene = None
    if new_map_v:
        st.session_state.draw_map_v += 1


# --------------------------------------------------------------------------
# Shared results rendering - used by both the single-site "Run & results"
# tab and the "Analyse uploaded data" tab, so both present a results table
# identically rather than maintaining two copies of the same metrics/plot
# code that could quietly drift apart.
# --------------------------------------------------------------------------

def render_results_summary(combined: pd.DataFrame, key_prefix: str, download_filename: str):
    """Renders the summary metrics, time-series plot, CSV download button,
    and plot-click selection for an already-deduped results DataFrame (the
    output of aggregate.prefer_sentinel_on_shared_dates, or an uploaded CSV
    in that same shape - date/sensor/status/... columns). `key_prefix`
    keeps this tab's Streamlit widget keys distinct from any other tab that
    also calls this function in the same run.

    If `combined` has a 'site' column with more than one distinct value
    (e.g. a batch run's combined CSV, single or re-uploaded), a selectbox
    lets the user narrow the view to one site or see every site combined -
    single-site results (no 'site' column) skip that control entirely.

    Returns (view, selected) - `view` is whatever subset of `combined` the
    site filter narrowed to (== combined if there's no filter or "All
    sites" is chosen), and `selected` is the pandas Series for the
    plot-clicked point, or None if nothing's been clicked yet - callers
    that can offer a scene preview (i.e. have an actual site geometry to
    fetch imagery for) use `selected` to drive that; callers that can't
    (e.g. arbitrary uploaded data with no corresponding site loaded) can
    just ignore it.
    """
    view = combined
    if "site" in combined.columns and combined["site"].nunique() > 1:
        site_choice = st.selectbox(
            "Site", ["All sites combined"] + sorted(combined["site"].dropna().unique().tolist()),
            key=f"{key_prefix}_site_filter",
        )
        if site_choice != "All sites combined":
            view = combined[combined["site"] == site_choice]

    method_choice = st.radio(
        "Classify using",
        ["Combined (open if either method connects)", "NDWI only", "fmask only"],
        horizontal=True,
        key=f"{key_prefix}_method",
        help=(
            "Combined (the default) calls a scene open if either method finds a "
            "connected path - this is what feeds the site-level statistics normally. "
            "NDWI/fmask only shows just that one method's own result, ignoring "
            "whether the other method agrees - useful for comparing the two or for "
            "digging into a scene where they disagree."
        ),
    )
    status_col = {
        "Combined (open if either method connects)": "status",
        "NDWI only": "status_ndwi",
        "fmask only": "status_fmask",
    }[method_choice]

    counts = aggregate.summary_counts(view, status_col=status_col)
    prop_closed = aggregate.mean_monthly_proportion_closed(view, status_col=status_col)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Open scenes", counts["n_open"])
    m2.metric("Closed scenes", counts["n_closed"])
    m3.metric("Indeterminate", counts["n_indeterminate"])
    m4.metric(
        "Mean monthly % closed",
        f"{prop_closed * 100:.1f}%" if prop_closed is not None else "n/a",
    )

    # Combined values are "open"/"closed"/"indeterminate"; the per-method columns
    # use "TRUE"/"FALSE"/"indeterminate" - one map covers both vocabularies.
    status_y = {"closed": 0, "FALSE": 0, "indeterminate": 0.5, "open": 1, "TRUE": 1}
    fig = go.Figure()
    for sensor, symbol, color in [("landsat", "circle-open", PASTEL["landsat"]), ("sentinel2", "circle", PASTEL["sentinel"])]:
        sub = view[view["sensor"] == sensor]
        if len(sub) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub[status_col].map(status_y),
                mode="markers",
                name=sensor,
                marker=dict(symbol=symbol, size=10, color=color),
                customdata=np.stack([sub["date"].dt.strftime("%Y-%m-%d"), sub["sensor"]], axis=-1),
                hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
                text=sub[status_col],
            )
        )
    fig.update_yaxes(tickvals=[0, 0.5, 1], ticktext=["closed", "indeterminate", "open"], range=[-0.2, 1.2])
    fig.update_layout(height=420, margin=dict(t=20, b=20), clickmode="event+select")

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "Download results as CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name=download_filename,
            mime="text/csv",
            key=f"{key_prefix}_download",
        )

    event = st.plotly_chart(fig, key=f"{key_prefix}_ts_plot", on_select="rerun", use_container_width=True)

    selected = None
    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        cd = pt.get("customdata")
        if cd:
            sel_date = pd.Timestamp(cd[0])
            sel_sensor = cd[1]
            match = view[(view["date"] == sel_date) & (view["sensor"] == sel_sensor)]
            if len(match) > 0:
                selected = match.iloc[0]

    return view, selected


# --------------------------------------------------------------------------
# Sidebar: site name + save/load
# --------------------------------------------------------------------------

st.sidebar.title("Estuary Mouth Monitor")

st.session_state.site_name = st.sidebar.text_input("Site name", value=st.session_state.site_name)

# Site-layer save/load - browser upload/download rather than a native OS
# folder dialog or on-disk Shapefiles (see modules/region.py's docstring
# for why: this app runs on a server with no access to your computer's
# filesystem). "Download" packages the current ROI/lines/structures into
# one JSON file; "Upload" reads a previously-downloaded file back in.
st.sidebar.markdown("**Save site layers**")
can_save = current_draw_step() == "ready"
if can_save:
    _site_for_download = SiteLayers(
        name=st.session_state.site_name,
        roi=st.session_state.roi_gdf,
        lines=st.session_state.lines_gdf,
        structures=st.session_state.structures_gdf,
        sandbars=st.session_state.sandbars_gdf,
    )
    save_dl_col, save_queue_col = st.sidebar.columns(2)
    with save_dl_col:
        st.download_button(
            "Download (.json)",
            data=_site_for_download.to_json_bytes(),
            file_name=f"{st.session_state.site_name}_site.json",
            mime="application/json",
            help="Saves the current ROI, inside/outside lines, structures and sandbars to a "
                 "single file on your own computer - upload it below any time to restore this "
                 "exact site without redrawing.",
        )
    with save_queue_col:
        if st.button("Add to batch queue"):
            st.session_state.site_queue = [s for s in st.session_state.site_queue if s.name != _site_for_download.name]
            st.session_state.site_queue.append(_site_for_download)
            n_queued = len(st.session_state.site_queue)
            # Queueing immediately clears the way for the next site: this
            # one stays visible (faded) on the draw tab's map via
            # site_queue above, and the confirm buttons reset to red for a
            # fresh blank site.
            reset_current_site()
            st.sidebar.success(f"Added - {n_queued} site(s) queued. Ready to draw the next site.")
            st.rerun()
else:
    st.sidebar.caption("Complete all five steps in tab 1 (region, lines, structures, sandbars) first to enable saving.")

st.sidebar.markdown("**Load site layers**")
uploaded_site = st.sidebar.file_uploader(
    "Upload a previously downloaded site-layers .json file", type=["json"], key="site_upload",
)
if uploaded_site is not None and st.sidebar.button("Load uploaded site layers"):
    try:
        loaded = SiteLayers.from_json_bytes(uploaded_site.getvalue())
        st.session_state.roi_gdf = loaded.roi
        st.session_state.lines_gdf = loaded.lines
        st.session_state.structures_gdf = loaded.structures
        st.session_state.sandbars_gdf = loaded.sandbars
        update_last_site_center(loaded.roi)
        st.session_state.structures_decided = True  # loading implies the decision was already made
        st.session_state.sandbars_decided = True
        st.session_state.site_name = loaded.name
        st.session_state.results_df = None
        st.session_state.clear_sky_references = {}
        st.session_state.preview_scene = None
        st.session_state.draw_map_v += 1
        st.sidebar.success(f"Loaded '{loaded.name}'.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))

if st.sidebar.button("Start a new (blank) site"):
    reset_current_site()
    st.rerun()

st.sidebar.markdown(f"**Sites queued for batch ({len(st.session_state.site_queue)})**")
st.sidebar.caption(
    "Draw a site in tab 1 and click 'Add to batch queue' above - it's added here and the map "
    "clears (faded) for the next site automatically. Nothing here is saved to the server - it "
    "only lives for this browser session, so download the batch's combined/per-site CSVs (tab 3) "
    "before closing the tab."
)
if st.session_state.site_queue:
    for i, qsite in enumerate(st.session_state.site_queue):
        qcol1, qcol2 = st.sidebar.columns([4, 1])
        qcol1.write(qsite.name)
        if qcol2.button("✕", key=f"remove_queued_{i}"):
            st.session_state.site_queue.pop(i)
            st.rerun()

with st.sidebar.expander("Bulk-add sites from JSON files", expanded=False):
    bulk_uploads = st.file_uploader(
        "Upload several previously-downloaded site-layers .json files at once",
        type=["json"], accept_multiple_files=True, key="bulk_site_upload",
    )
    if bulk_uploads and st.button("Add all uploaded sites to the batch queue"):
        added = []
        existing_names = {s.name for s in st.session_state.site_queue}
        for f in bulk_uploads:
            try:
                loaded_site = SiteLayers.from_json_bytes(f.getvalue())
                if loaded_site.name not in existing_names:
                    st.session_state.site_queue.append(loaded_site)
                    existing_names.add(loaded_site.name)
                    added.append(loaded_site.name)
            except Exception as e:
                st.warning(f"Could not load '{f.name}': {e}")
        st.success(f"Added {len(added)} site(s) to the batch queue.")
        st.rerun()

with st.sidebar.expander("About the classification", expanded=False):
    st.markdown(
        """
        For each scene: an NDWI water mask and the image's own oa_fmask
        water class (5) are each tested for a connected path between the
        inside/outside lines (8-connected neighbours). **Open** if either
        method finds a path. **Closed** if neither does and nothing is
        blocking the view. **Indeterminate** if cloud/no-data sits
        somewhere a path could plausibly have run through. Pixels under any
        drawn structure polygon (bridges, causeways) are always treated as
        passable water. Pixels under a drawn sandbar polygon are exempted
        from the temporal-anomaly check only (if enabled) - a sandbar
        appearing or disappearing between scenes isn't treated as
        cloud/haze contamination, so its water/land call is never demoted
        to indeterminate purely for looking different from its own history.
        """
    )


# --------------------------------------------------------------------------
# Main layout: drawing workflow tabs + run/results tab
# --------------------------------------------------------------------------

tab_home, tab_draw, tab_run, tab_batch, tab_upload = st.tabs(
    [
        "Home", "1. Draw & save site layers", "2. Run & results",
        "3. Batch run (multiple sites)", "4. Analyse uploaded data",
    ]
)

with tab_home:
    st.title("Estuary Mouth Monitor")
    st.markdown(
        """
        This app detects whether an estuary mouth is **open or closed** using
        satellite imagery, and lets you view and analyse how often it closes
        over time.

        It works by fetching every available Landsat and Sentinel-2 scene
        over a chosen estuary from [Digital Earth Australia](https://www.dea.ga.gov.au/)
        and testing, on each one, whether there's a connected path of water
        between a point just inside the mouth and a point just outside it.
        Run it over a period of months or years and you get a time series of
        open/closed/indeterminate calls - and from that, statistics like the
        proportion of time the mouth has been closed.
        """
    )

    if os.path.exists(MURRAY_MOUTH_IMAGE):
        st.image(
            MURRAY_MOUTH_IMAGE,
            caption=(
                "The Murray River mouth, South Australia - a Landsat/Sentinel-2 scene over the "
                "Coorong. The blue box is an example region of interest drawn around the mouth, "
                "the same first step you'll do for your own site in tab 1 below."
            ),
        )

    st.subheader("How to use it")
    st.markdown(
        """
        Work through the numbered tabs above in order, left to right:

        1. **Draw & save site layers** - draw everything for one site on a
           single map, confirming each layer in turn with the buttons beside
           it (they turn from red to green as you go): region of interest,
           inside line, outside line, structures, then sandbars.
           Region/lines are required; structures and sandbars are optional -
           just press their confirm button with nothing drawn if there
           aren't any at this site. Once all five are green, click "Add to
           batch queue" in the sidebar: this site's layers stay on the map
           faded out, its name (auto-generated from the ROI's centroid, e.g.
           `site_38.1234S_140.5678E`) is added to the batch queue, and the
           buttons reset to red so you can draw the next site straight away.
        2. **Run & results** - pick a date range and maximum cloud cover,
           then run the analysis. You'll get a time series plot of
           open/closed/indeterminate status, summary statistics (including
           the mean monthly proportion of time closed), and a scene preview
           where you can click any point on the plot to see exactly what the
           satellite image and classification looked like on that date.
        3. **Batch run (multiple sites)** - draw and queue several sites in
           tab 1 (see "Sites queued for batch" in the sidebar), then run the
           same analysis across all of them in one go, downloading a
           combined CSV and a per-site CSV zip at the end.
        4. **Analyse uploaded data** - re-upload a results CSV this app
           previously exported (from tab 2 or tab 3) to see the same
           summary statistics and plot again without re-fetching anything
           from DEA - handy for revisiting old results or combining CSVs
           from separate sessions.

        A few other things worth knowing about, in the sidebar:

        - **Save / load site layers** lets you download your drawn ROI,
          lines, structures and sandbars as a single file, and re-upload it
          later to pick up an existing site without redrawing. Auto-names
          the site from its ROI's centroid (e.g. `site_38.1234S_140.5678E`)
          when you confirm the region in tab 1.
        - **Sites queued for batch** lists every site you've added this
          session (one at a time via "Add to batch queue", or several at
          once via "Bulk-add sites from JSON files"), ready for tab 3's
          batch run - they also stay visible, faded, on tab 1's map as you
          draw more sites. Nothing here is saved on the server - it only
          lasts for this browser session.
        - **About the classification** (further down the sidebar) explains
          the open/closed/indeterminate logic in more detail.
        - The "Advanced" sections in tab 2 (cloud-edge buffer, temporal
          anomaly detection) are optional tuning for tricky sites where
          cloud or haze is causing false results - the defaults work well
          for most estuaries, so it's fine to leave them alone starting out.
        """
    )

with tab_draw:
    step = current_draw_step()
    confirmed = draw_step_confirmed()

    if step == "ready":
        st.success(
            "✓ All layers confirmed for this site - click **Add to batch queue** in the sidebar "
            "to queue it and start the next one."
        )
    else:
        st.write(
            "Draw each layer on the map below, then press its confirm button on the right (it turns "
            "green once confirmed). Work through them in order: region of interest, inside line, "
            "outside line, structures, sandbars. Structures and sandbars are optional - press "
            "their confirm button with nothing drawn if there aren't any at this site."
        )

    STEP_INSTRUCTIONS = {
        "roi": "Use the polygon or rectangle tool to draw a region around the estuary mouth. This "
               "defines both the area rasters are fetched for and clipped to.",
        "inside": "Use the polyline tool to draw a line crossing the water on the river ('inside') "
                  "side of the mouth.",
        "outside": "Use the polyline tool to draw a line crossing the water on the ocean ('outside') "
                   "side of the mouth.",
        "structures": "Optional: draw polygons over any structures (bridges, causeways) that cross "
                      "the estuary. Pixels under these are always treated as passable water.",
        "sandbars": "Optional: draw polygons over any sandbar that can appear or disappear over "
                    "time. Unlike structures, sandbar pixels are **not** forced to water - they "
                    "keep whatever the ordinary NDWI/fmask classification finds. The only thing "
                    "this changes is the temporal-anomaly check (tab 2's Advanced section, if "
                    "enabled): pixels under a drawn sandbar are exempted from it entirely, rather "
                    "than being wrongly flagged as indeterminate right where the mouth is most "
                    "likely to open or close.",
    }
    if step in STEP_INSTRUCTIONS:
        st.caption(STEP_INSTRUCTIONS[step])

    map_col, btn_col = st.columns([5, 1])

    with map_col:
        if st.session_state.roi_gdf is not None:
            center, zoom = map_center_from_roi(), 14
        elif st.session_state.last_site_center is not None:
            # No in-progress ROI right now (e.g. just queued a site) - stay
            # put on the last estuary that was defined rather than jumping
            # back out to the whole-of-Australia view.
            center, zoom = st.session_state.last_site_center, 14
        else:
            center, zoom = AUSTRALIA_CENTER, 5
        m = satellite_map(center=center, zoom=zoom)

        # Previously-queued sites this session, faded, so they stay visible
        # as spatial context while drawing the next one on the same map.
        for saved in st.session_state.site_queue:
            add_context_layers(
                m, roi=saved.roi, lines=saved.lines, structures=saved.structures,
                sandbars=saved.sandbars, alpha=0.3,
            )

        # The current (in-progress) site's already-confirmed layers, full
        # opacity, as non-editable context for whichever step is active now.
        add_context_layers(
            m,
            roi=st.session_state.roi_gdf if step != "roi" else None,
            lines=st.session_state.lines_gdf,
            structures=st.session_state.structures_gdf if step in ("sandbars", "ready") else None,
            sandbars=st.session_state.sandbars_gdf if step == "ready" else None,
        )

        draw_opts = {"polygon": False, "polyline": False, "rectangle": False, "circle": False, "marker": False, "circlemarker": False}
        if step == "roi":
            draw_opts["polygon"] = True
            draw_opts["rectangle"] = True
        elif step in ("inside", "outside"):
            draw_opts["polyline"] = True
        elif step in ("structures", "sandbars"):
            draw_opts["polygon"] = True
            draw_opts["rectangle"] = True

        if step != "ready":
            Draw(export=False, draw_options=draw_opts, edit_options={"edit": True, "remove": True}).add_to(m)

        map_data = st_folium(m, key=f"draw_map_{st.session_state.draw_map_v}", height=700, use_container_width=True)

        drawings = (map_data or {}).get("all_drawings") or []
        if step == "roi" or step in ("structures", "sandbars"):
            candidates = [f for f in drawings if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
        elif step in ("inside", "outside"):
            candidates = [f for f in drawings if f["geometry"]["type"] in ("LineString", "MultiLineString")]
        else:
            candidates = []

    with btn_col:
        st.caption("Confirm in order:")
        for s in DRAW_STEPS:
            is_confirmed = confirmed[s]
            is_active = s == step
            css_key = f"confirm_btn_{s}"
            colour = "#8FCB9B" if is_confirmed else "#E8A0A0"
            st.markdown(
                f"""<style>
                .st-key-{css_key} button {{
                    background-color: {colour} !important;
                    border-color: {colour} !important;
                    color: #1a1a1a !important;
                }}
                </style>""",
                unsafe_allow_html=True,
            )
            if is_confirmed:
                label = f"✓ {DRAW_STEP_LABELS[s]}"
            else:
                label = DRAW_STEP_LABELS[s]
            with st.container(key=css_key):
                clicked = st.button(label, key=f"btn_{s}", use_container_width=True, disabled=not (is_active or is_confirmed))
            if clicked:
                if is_confirmed:
                    redraw_draw_step(s)
                elif is_active:
                    confirm_draw_step(s, candidates)

with tab_run:
    ready = st.session_state.roi_gdf is not None and st.session_state.lines_gdf is not None
    if not ready:
        st.warning("Complete tab 1 (region + lines) before running an analysis.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            start = st.date_input("Start date", value=LANDSAT5_START, min_value=LANDSAT5_START, max_value=date.today())
        with c2:
            end = st.date_input("End date", value=date.today(), min_value=LANDSAT5_START, max_value=date.today())
        with c3:
            max_cloud = st.slider("Max cloud cover (%) per scene", min_value=0, max_value=100, value=20)

        with st.expander("Advanced: cloud-edge buffer", expanded=False):
            st.caption(
                "Pixels right at a cloud's edge (haze, thin cirrus, mixed cloud/land pixels) often "
                "aren't flagged as cloud or shadow by fmask, but are still contaminated enough to "
                "throw off the water/land call - which can show up as a thin false 'dry' sliver right "
                "where a cloud edge crosses a channel, wrongly turning an open scene into 'closed'. "
                "This dilates the cloud mask by the pixel counts below and treats that ring as "
                "indeterminate instead of trusting the (unreliable) land/water call there. Set to 0 "
                "to disable and reproduce the original behaviour."
            )
            buf_c1, buf_c2 = st.columns(2)
            with buf_c1:
                st.session_state.cloud_buffer_s2 = st.number_input(
                    "Sentinel-2 buffer (pixels, 10m each)", min_value=0, max_value=100,
                    value=int(st.session_state.cloud_buffer_s2),
                )
            with buf_c2:
                st.session_state.cloud_buffer_ls = st.number_input(
                    "Landsat buffer (pixels, 30m each)", min_value=0, max_value=20,
                    value=int(st.session_state.cloud_buffer_ls),
                )
        # Kept in session_state (not just a local variable) so the batch-run
        # tab can reuse the same tuned values without duplicating this UI -
        # cloud-buffer/temporal-anomaly settings are analysis-wide tuning,
        # not really specific to a single run.
        cloud_buffer_px = {"sentinel2": st.session_state.cloud_buffer_s2, "landsat": st.session_state.cloud_buffer_ls}

        with st.expander("Advanced: temporal cloud/haze anomaly detection", expanded=False):
            st.caption(
                "The cloud-edge buffer above only ever grows outward from pixels fmask itself "
                "already flagged as cloud - it can't help with thin, dappled or broken cloud fmask "
                "misses entirely, which shows up as ordinary-looking 'clear' pixels with no cloud "
                "flag anywhere nearby. This is a different check: it compares each pixel's blue-band "
                "brightness in a scene against how that *same* pixel has looked on this site's "
                "clearest other dates, and flags it as probably cloud/haze-contaminated if it's "
                "anomalously bright - regardless of what fmask says. Needs a handful of clear "
                "historical observations per pixel to build a reliable reference, and the threshold "
                "below will likely need calibrating against real scenes (use the scene preview's "
                "mask overlay below, after a run, to check - flagged pixels show up as a new colour "
                "class there)."
            )
            st.session_state.enable_temporal_anomaly = st.checkbox(
                "Enable temporal anomaly detection", value=st.session_state.enable_temporal_anomaly,
            )
            ta_c1, ta_c2, ta_c3 = st.columns(3)
            with ta_c1:
                st.session_state.temporal_anomaly_threshold = st.number_input(
                    "Brightness threshold (blue-band DN, x10000 reflectance)",
                    min_value=0.0, max_value=5000.0, value=float(st.session_state.temporal_anomaly_threshold), step=50.0,
                    disabled=not st.session_state.enable_temporal_anomaly,
                    help="How far above a pixel's own clear-sky reference its blue-band value has to sit "
                         "before it's flagged. Lower = catches more haze but risks false-flagging naturally "
                         "variable pixels (e.g. turbid water, wet sand at different tides).",
                )
            with ta_c2:
                st.session_state.temporal_anomaly_percentile = st.number_input(
                    "Reference percentile", min_value=0.0, max_value=50.0,
                    value=float(st.session_state.temporal_anomaly_percentile), step=5.0,
                    disabled=not st.session_state.enable_temporal_anomaly,
                    help="The low percentile of each pixel's clear/water-observation history used as its "
                         "'normal' brightness. Lower is more conservative (closer to the single clearest "
                         "observation) but noisier with few observations.",
                )
            with ta_c3:
                st.session_state.temporal_anomaly_min_obs = st.number_input(
                    "Min. clear observations required", min_value=1, max_value=50,
                    value=int(st.session_state.temporal_anomaly_min_obs), step=1,
                    disabled=not st.session_state.enable_temporal_anomaly,
                    help="Pixels with fewer clear/water observations than this across the whole date range "
                         "get no reference at all, and are never flagged by this check (falls back to "
                         "fmask/buffer behaviour only).",
                )

        require_full_coverage = st.checkbox(
            "Only use scenes that fully cover the drawn region (recommended)",
            value=True,
            help=(
                "Skips scenes where the ROI straddles two tiles collected on different days, or "
                "sits at the edge of a swath - these otherwise usually just come out indeterminate "
                "anyway, since part of the region has no data. Turning this off keeps every scene "
                "regardless of coverage, at the cost of more indeterminate results and slower runs."
            ),
        )

        run_clicked = st.button("Run analysis", type="primary")

        if run_clicked:
            site = SiteLayers(
                name=st.session_state.site_name,
                roi=st.session_state.roi_gdf,
                lines=st.session_state.lines_gdf,
                structures=st.session_state.structures_gdf,
                sandbars=st.session_state.sandbars_gdf,
            )
            problems = site.validate()
            if problems:
                st.error("Cannot run: " + "; ".join(problems))
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def progress_cb(done, total, message):
                    status_text.write(message)
                    if total:
                        progress_bar.progress(min(done / total, 1.0))

                with st.spinner("Fetching and analysing scenes..."):
                    try:
                        records, clear_sky_references = fetch.run_site_analysis(
                            site=site,
                            start_date=start.isoformat(),
                            end_date=end.isoformat(),
                            max_cloud=max_cloud,
                            products_json_path=PRODUCTS_JSON,
                            min_roi_coverage=fetch.FULL_COVERAGE_THRESHOLD if require_full_coverage else 0,
                            progress_cb=progress_cb,
                            cloud_buffer_px=cloud_buffer_px,
                            enable_temporal_anomaly=st.session_state.enable_temporal_anomaly,
                            temporal_anomaly_threshold=st.session_state.temporal_anomaly_threshold,
                            temporal_anomaly_percentile=st.session_state.temporal_anomaly_percentile,
                            temporal_anomaly_min_obs=st.session_state.temporal_anomaly_min_obs,
                        )
                        st.session_state.raw_records = records
                        st.session_state.results_df = aggregate.build_results_df(records)
                        st.session_state.clear_sky_references = clear_sky_references
                        st.session_state.preview_scene = None  # a fresh run invalidates any stale preview
                        status_text.write(f"Done - {len(records)} scenes processed.")
                    except Exception as e:
                        st.exception(e)

        df = st.session_state.results_df
        if df is not None and len(df) > 0:
            combined = aggregate.prefer_sentinel_on_shared_dates(df)
            combined, selected = render_results_summary(
                combined, key_prefix="run", download_filename=f"{st.session_state.site_name}_results.csv",
            )

            st.divider()
            st.subheader("Scene preview")
            if selected is None:
                st.caption("Click a point on the plot above to preview that scene.")
            else:
                # Reproject the confirmed lines into this scene's CRS for the overlay.
                # Rebuilt from session_state (rather than reusing the `site` variable from
                # the Run button branch) so this works even on reruns triggered purely by
                # clicking a plot point, without needing Run to have just been pressed.
                # Built up-front since it's also needed to fetch the scene on demand below.
                site_for_preview = SiteLayers(
                    name=st.session_state.site_name,
                    roi=st.session_state.roi_gdf,
                    lines=st.session_state.lines_gdf,
                    structures=st.session_state.structures_gdf,
                    sandbars=st.session_state.sandbars_gdf,
                )

                # Only the single most-recently-viewed scene is ever kept in
                # memory (st.session_state.preview_scene), never written to
                # disk - see modules/fetch.py's docstring for why (a
                # server-side cache shared across every visitor risks one
                # user seeing another user's scene). Selecting a different
                # point drops the old scene's arrays entirely (they're just
                # overwritten here, so Python garbage-collects them) and
                # re-fetches from DEA; re-clicking the same point reuses
                # what's already in memory for this session.
                preview_key = (site_for_preview.name, selected["sensor"], selected["date"].strftime("%Y-%m-%d"))
                cached_preview = st.session_state.preview_scene
                fetch_diagnostics = None
                if cached_preview is not None and cached_preview["key"] == preview_key:
                    scene_data = (cached_preview["bands"], cached_preview["transform"], cached_preview["crs"])
                else:
                    scene_data = None
                    with st.spinner(f"Downloading the {selected['sensor']} scene for {selected['date'].date()} from DEA..."):
                        try:
                            bands, transform, crs, fetch_diagnostics = fetch.fetch_single_scene(
                                site_for_preview, selected["date"], selected["sensor"], PRODUCTS_JSON,
                            )
                            scene_data = (bands, transform, crs)
                            st.session_state.preview_scene = {
                                "key": preview_key, "bands": bands, "transform": transform, "crs": crs,
                            }
                        except Exception as e:
                            st.error(f"Could not fetch this scene from DEA: {e}")

                clear_sky_reference = st.session_state.clear_sky_references.get(
                    fetch.SENSOR_PRODUCT.get(selected["sensor"])
                )

                # Surface what was actually fetched - a near-blank preview is much
                # easier to diagnose with this in front of you than by guessing.
                if fetch_diagnostics is not None:
                    vpf = fetch_diagnostics.get("valid_pixel_frac")
                    if vpf is not None and vpf < 0.5:
                        st.warning(
                            f"Only {vpf:.0%} of this scene's pixels are valid (non-no-data) "
                            "within the ROI after fetching - this looks like a genuine "
                            "low-coverage or missing-tile issue for this date, not just a "
                            "rendering problem. Details below."
                        )
                    with st.expander(
                        f"Fetch details ({fetch_diagnostics.get('source')}"
                        + (f" - valid pixels {vpf:.0%})" if vpf is not None else ")")
                    ):
                        st.json(fetch_diagnostics)

            if selected is not None and scene_data is not None:
                bg_c1, bg_c2 = st.columns([1, 2])
                with bg_c1:
                    bg_choice = st.radio(
                        "Background", ["NDWI (continuous)", "fmask (categorical)"], horizontal=True,
                        help=(
                            "NDWI shows the continuous index each pixel gets tested against (>0 = "
                            "water). fmask shows the image's own oa_fmask classification directly - "
                            "the raw data the fmask connectivity method uses - as flat colour classes: "
                            "water, land, cloud, cloud shadow, snow (treated as water), no-data, and "
                            "outside the drawn ROI."
                        ),
                    )
                with bg_c2:
                    show_mask = bg_choice == "NDWI (continuous)" and st.checkbox(
                        "Show cloud / no-data mask overlay", value=True,
                        help=(
                            "Grey = no-data, lavender = cloud, tan = inside a cloud's buffer zone, and "
                            "orange = a temporal brightness anomaly are all excluded from the water/land "
                            "call (these are what actually drive an indeterminate result) - unless NDWI or "
                            "fmask already calls that cell water, in which case it's shown in teal instead "
                            "and NOT excluded (a nearby cloud doesn't override an already-positive water "
                            "call). Mint = 'likely broken water' (fmask misreads wave/foam as snow, "
                            "already treated as water). Sky blue = cloud shadow is shown for reference only "
                            "- not excluded. Pale yellow = outside the drawn ROI. Hover any tinted cell for "
                            "its exact category."
                        ),
                    )
                bands, transform, crs = scene_data
                st.download_button(
                    "Download this scene as GeoTIFF",
                    data=fetch.write_scene_geotiff_bytes(bands, transform, crs),
                    file_name=f"{site_for_preview.name}_{selected['sensor']}_{selected['date'].date()}.tif",
                    mime="image/tiff",
                    help="Saves this scene straight to your own computer - the server never keeps a copy.",
                )
                sensor, nir_band = connectivity.detect_sensor_and_nir_band(bands.keys())
                ndwi = connectivity.calc_ndwi(bands["nbart_green"], bands[nir_band])
                fmask = bands["oa_fmask"]

                # site_for_preview was already built above (before the fetch), since it's
                # needed either way; just reproject it into this scene's CRS here.
                scene_crs = crs.to_epsg()
                inside_px, outside_px = [], []
                for gdf, bucket in [(site_for_preview.inside_line(scene_crs), inside_px), (site_for_preview.outside_line(scene_crs), outside_px)]:
                    for geom in gdf.geometry:
                        for x, y in geom.coords:
                            r, c = rowcol(transform, x, y)
                            bucket.append((c, r))

                fig2 = go.Figure()
                # Colorbar pinned to the far right with a fixed length/thickness so
                # it doesn't stretch the full height of the plot; the inside/outside
                # legend is placed above the plot (horizontal) so the two never overlap.
                if bg_choice == "NDWI (continuous)":
                    fig2.add_trace(
                        go.Heatmap(
                            z=ndwi, colorscale="RdBu", zmid=0, showscale=True, name="NDWI",
                            colorbar=dict(title="NDWI", thickness=15, len=0.9, x=1.02),
                        )
                    )
                else:
                    # Raw oa_fmask classes, mapped straight to colour with no NDWI
                    # involved - codes: 0 nodata, 1 land, 2 cloud, 3 cloud shadow,
                    # 4 snow (treated as water), 5 water, 6 outside the drawn ROI
                    # (the clip-induced nodata sentinel - see unknown_from_fmask()).
                    fmask_codes = np.where(np.isin(fmask, [0, 1, 2, 3, 4, 5]), fmask, 6).astype(float)
                    fmask_colors = [
                        PASTEL["nodata"], PASTEL["land"], PASTEL["cloud"],
                        PASTEL["cloud_shadow"], PASTEL["snow_water"], PASTEL["water"], PASTEL["outside_roi"],
                    ]
                    fig2.add_trace(
                        go.Heatmap(
                            z=fmask_codes, zmin=-0.5, zmax=6.5, colorscale=discrete_colorscale(fmask_colors),
                            showscale=False, name="fmask",
                        )
                    )
                if show_mask:
                    # Categorical overlay - mirrors exactly what
                    # connectivity.py's apply_cloud_buffer/apply_temporal_
                    # anomaly actually do to the real classification (see
                    # those functions' docstrings), rather than just raw
                    # fmask categories, so this diagnostic view can be
                    # trusted to show what the algorithm is really doing:
                    #   1 = no-data, 2 = cloud, 3 = cloud shadow (reference
                    #   only, not excluded), 4 = outside the drawn ROI,
                    #   5 = cloud buffer zone that ISN'T already water,
                    #   6 = temporal brightness anomaly that ISN'T already
                    #   water, 7 = "likely broken water" (fmask misreads
                    #   wave/foam as snow, oa_fmask==4 - already forced to
                    #   water, so never excluded regardless of brightness),
                    #   8 = water (NDWI or fmask) that happens to sit inside
                    #   a cloud buffer or anomaly zone - shown distinctly so
                    #   it's clear this is NOT excluded, unlike 5/6.
                    # NaN cells render transparent, so only flagged pixels
                    # show through over the NDWI image underneath.
                    _, fmask_updated_here = connectivity.build_updated_fmask(fmask)
                    ndwi_water_here, _ = connectivity.build_ndwi_water_mask(fmask, ndwi)
                    fmask_water_here, _ = connectivity.build_fmask_water_mask(fmask, fmask_updated_here, ndwi)
                    water_strict_here = ndwi_water_here | fmask_water_here  # matches "combined" open/closed logic
                    snow_mask_here = fmask == 4

                    buffer_px_here = cloud_buffer_px.get(sensor, connectivity.DEFAULT_CLOUD_BUFFER_PX.get(sensor, 0))
                    cloud_buffer_mask_here = connectivity.build_cloud_buffer_mask(fmask, buffer_px_here)

                    anomaly_mask_here = None
                    if clear_sky_reference is not None and "nbart_blue" in bands:
                        anomaly_mask_here = connectivity.build_temporal_anomaly_mask(
                            bands["nbart_blue"], clear_sky_reference, st.session_state.temporal_anomaly_threshold,
                        )
                        sandbars_for_overlay = site_for_preview.sandbars_reprojected(scene_crs)
                        sandbar_mask_here = connectivity.build_structure_mask(fmask.shape, transform, sandbars_for_overlay)
                        if sandbar_mask_here is not None:
                            # Mirror process_scene()'s sandbar exemption exactly, so this
                            # diagnostic overlay shows the same thing the real analysis does.
                            anomaly_mask_here = anomaly_mask_here & ~sandbar_mask_here

                    mask_codes = np.full(fmask.shape, np.nan)
                    mask_labels = np.full(fmask.shape, "", dtype=object)

                    def _paint(selector, code, label):
                        mask_codes[selector] = code
                        mask_labels[selector] = label

                    _paint(fmask == 0, 1, "No-data")
                    _paint(fmask == 2, 2, "Cloud")
                    _paint(fmask == 3, 3, "Cloud shadow (reference only - not excluded)")
                    _paint(~np.isin(fmask, [0, 1, 2, 3, 4, 5]), 4, "Outside the drawn ROI")
                    _paint(
                        cloud_buffer_mask_here & ~water_strict_here & ~snow_mask_here, 5,
                        "Cloud buffer zone - not water (excluded)",
                    )
                    if anomaly_mask_here is not None:
                        _paint(
                            anomaly_mask_here & ~water_strict_here & ~snow_mask_here, 6,
                            "Temporal brightness anomaly - not water (excluded)",
                        )
                    _paint(
                        snow_mask_here, 7,
                        'Likely broken water (fmask misreads wave/foam as "snow" - already treated as water)',
                    )
                    protected_here = cloud_buffer_mask_here | (
                        anomaly_mask_here if anomaly_mask_here is not None else np.zeros_like(fmask, dtype=bool)
                    )
                    _paint(
                        water_strict_here & ~snow_mask_here & protected_here, 8,
                        "Water (NDWI/fmask) near cloud or flagged as an anomaly - NOT excluded",
                    )

                    mask_colors = [
                        PASTEL["nodata"], PASTEL["cloud"], PASTEL["cloud_shadow"], PASTEL["outside_roi"],
                        PASTEL["cloud_buffer"], PASTEL["temporal_anomaly"], PASTEL["snow_water"],
                        PASTEL["protected_water"],
                    ]
                    if np.any(~np.isnan(mask_codes)):
                        fig2.add_trace(
                            go.Heatmap(
                                z=mask_codes, zmin=0.5, zmax=8.5, colorscale=discrete_colorscale(mask_colors),
                                showscale=False, opacity=0.85, hoverinfo="text", text=mask_labels, name="mask",
                            )
                        )
                if inside_px:
                    xs, ys = zip(*inside_px)
                    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=PASTEL["inside"], width=3), name="inside"))
                if outside_px:
                    xs, ys = zip(*outside_px)
                    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=PASTEL["outside"], width=3), name="outside"))
                fig2.update_yaxes(autorange="reversed", scaleanchor="x")
                fig2.update_layout(
                    height=550,
                    margin=dict(t=90),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    title=dict(
                        text=(
                            f"{selected['date'].date()} - {selected['sensor']} - status: {selected['status']} "
                            f"(NDWI: {selected['status_ndwi']}, fmask: {selected['status_fmask']})"
                        ),
                        y=0.98,
                    ),
                )
                st.plotly_chart(fig2, use_container_width=True)
                if bg_choice == "NDWI (continuous)" and show_mask:
                    st.caption(
                        "Mask overlay - grey: no-data · lavender: cloud · pale yellow: outside the "
                        "drawn ROI · sky blue: cloud shadow (shown for reference only - still classified "
                        "via NDWI, not excluded) · tan: inside a cloud's buffer zone · orange: temporal "
                        "brightness anomaly (see the 'Advanced' sections above for both) - tan and orange "
                        "cells are excluded from the water/land call *only* when NDWI/fmask don't already "
                        "call them water. Mint: 'likely broken water' - fmask misreads wave/foam as "
                        "\"snow\" (oa_fmask==4); this is already forced to water everywhere in this app, "
                        "never excluded. Teal: water (NDWI or fmask) that happens to sit inside a cloud "
                        "buffer or anomaly zone - shown distinctly so it's clear this is water, not "
                        "excluded, despite the nearby cloud/anomaly flag."
                    )
                    if clear_sky_reference is None:
                        st.caption(
                            "No clear-sky reference built for this site/sensor this session yet, so the "
                            "orange temporal-anomaly class can't be shown here - run an analysis with "
                            "'Enable temporal anomaly detection' checked first, which builds one from that "
                            "run's own scenes (kept in memory for this session only, never saved to disk)."
                        )
                elif bg_choice != "NDWI (continuous)":
                    st.caption(
                        "fmask classes - blue: water (== 5) · mint: snow (== 4, treated as water) · "
                        "sand: land/valid · lavender: cloud · sky blue: cloud shadow (classified via "
                        "NDWI, not excluded) · grey: no-data · pale yellow: outside the drawn ROI. "
                        "(The temporal brightness anomaly check isn't part of fmask, so it's not shown "
                        "on this background - switch to NDWI with the mask overlay on to see it.)"
                    )

                st.divider()
                diag_c1, diag_c2 = st.columns([2, 1])
                with diag_c1:
                    show_diag = st.checkbox(
                        "Show connectivity diagnostic (why open/closed/indeterminate)",
                        value=False,
                        help=(
                            "Recomputes this scene's connectivity check and shows which cells the "
                            "algorithm actually found reachable from the inside line, the outside "
                            "line, or both, when cloud/no-data cells are optimistically assumed to be "
                            "water. If a 'reaches both' patch shows up, the result should not be "
                            "'closed' for that method - a direct way to check a surprising result "
                            "instead of eyeballing the raster."
                        ),
                    )
                with diag_c2:
                    diag_variant = st.radio(
                        "Method", ["NDWI", "fmask"], horizontal=True,
                        disabled=not show_diag, label_visibility="collapsed", key="diag_variant",
                    )

                if show_diag:
                    inside_gdf = site_for_preview.inside_line(scene_crs)
                    outside_gdf = site_for_preview.outside_line(scene_crs)
                    structures_gdf = site_for_preview.structures_reprojected(scene_crs)
                    sandbars_gdf_preview = site_for_preview.sandbars_reprojected(scene_crs)
                    scene_result = connectivity.process_scene(
                        bands, transform, inside_gdf, outside_gdf, structures_gdf, sandbars_gdf_preview,
                        cloud_buffer_px=cloud_buffer_px,
                        clear_sky_reference=clear_sky_reference, temporal_anomaly_threshold=st.session_state.temporal_anomaly_threshold,
                    )
                    check = scene_result.ndwi_check if diag_variant == "NDWI" else scene_result.fmask_check

                    if check.optimistic is None:
                        st.info(
                            f"The {diag_variant} check already connects on the strict pass (status "
                            "TRUE) - there's no optimistic pass to show for this method."
                        )
                    else:
                        patch_r = check.optimistic.patch_raster
                        in_p, out_p, shared_p = (
                            check.optimistic.inside_patches,
                            check.optimistic.outside_patches,
                            check.optimistic.shared_patches,
                        )
                        diag_codes = np.full(patch_r.shape, np.nan)
                        diag_codes[np.isin(patch_r, list(in_p - shared_p))] = 1
                        diag_codes[np.isin(patch_r, list(out_p - shared_p))] = 2
                        diag_codes[np.isin(patch_r, list(shared_p))] = 3
                        diag_colors = [PASTEL["inside"], PASTEL["outside"], "#7CDB8A"]  # inside-only, outside-only, reaches-both

                        fig3 = go.Figure()
                        fig3.add_trace(go.Heatmap(z=ndwi, colorscale="Greys", showscale=False, opacity=0.45, hoverinfo="skip"))
                        if np.any(~np.isnan(diag_codes)):
                            fig3.add_trace(
                                go.Heatmap(
                                    z=diag_codes, zmin=0.5, zmax=3.5, colorscale=discrete_colorscale(diag_colors),
                                    showscale=False, opacity=0.85, hoverinfo="skip",
                                )
                            )
                        if inside_px:
                            xs, ys = zip(*inside_px)
                            fig3.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=PASTEL["inside"], width=3), name="inside line"))
                        if outside_px:
                            xs, ys = zip(*outside_px)
                            fig3.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=PASTEL["outside"], width=3), name="outside line"))
                        fig3.update_yaxes(autorange="reversed", scaleanchor="x")
                        fig3.update_layout(
                            height=550,
                            margin=dict(t=90),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                            title=dict(
                                text=(
                                    f"Optimistic pass ({diag_variant}, cloud/no-data assumed water) - "
                                    "blue: reaches inside line only · rose: reaches outside line only · "
                                    "green: reaches BOTH"
                                ),
                                y=0.98,
                            ),
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                        method_status = check.status  # this method's own "TRUE"/"FALSE"/"indeterminate"
                        if len(shared_p) > 0:
                            if method_status == "indeterminate":
                                st.info(
                                    f"A patch reaching both lines WAS found under the optimistic "
                                    f"{diag_variant} pass, while the strict pass (cloud/no-data NOT "
                                    f"assumed water) did not connect - that's exactly why {diag_variant}'s "
                                    "own status is **indeterminate**, not closed. This is expected "
                                    "behaviour, not a bug. Remember the scene's overall status (top of "
                                    "the preview above) combines NDWI and fmask - open if either is TRUE, "
                                    "closed only if **both** are FALSE, indeterminate otherwise - so check "
                                    "the other method's radio button too if its status is what's driving "
                                    "the overall result."
                                )
                            else:
                                st.warning(
                                    f"A patch reaching both lines WAS found under the optimistic "
                                    f"{diag_variant} pass, but this method's own status shows "
                                    f"'{method_status}' rather than TRUE or indeterminate - that looks "
                                    "like a genuine bug. Let me know and I'll dig into it."
                                )
                        else:
                            st.success(
                                f"No patch reaches both lines even treating all cloud/no-data as water "
                                f"under the {diag_variant} method - its own status is '{method_status}', "
                                "consistent with what the algorithm computed. The scene's overall status "
                                "(top of the preview above) combines NDWI and fmask, so check the other "
                                "method's radio button too if you want the full picture of why."
                            )

            with st.expander("Full results table"):
                st.dataframe(combined)


with tab_batch:
    st.subheader("Batch run across queued sites")
    if not st.session_state.site_queue:
        st.info(
            "No sites queued yet. Draw a site in tab 1 (its confirm buttons all turn green), then "
            "click 'Add to batch queue' in the sidebar to add it here, or use 'Bulk-add sites from "
            "JSON files' in the sidebar to add several previously-downloaded sites at once."
        )
    else:
        st.write(
            f"**{len(st.session_state.site_queue)} site(s) queued:** "
            + ", ".join(s.name for s in st.session_state.site_queue)
        )

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            b_start = st.date_input(
                "Start date", value=LANDSAT5_START, min_value=LANDSAT5_START, max_value=date.today(), key="batch_start",
            )
        with bc2:
            b_end = st.date_input(
                "End date", value=date.today(), min_value=LANDSAT5_START, max_value=date.today(), key="batch_end",
            )
        with bc3:
            b_max_cloud = st.slider(
                "Max cloud cover (%) per scene", min_value=0, max_value=100, value=20, key="batch_max_cloud",
            )

        b_require_full_coverage = st.checkbox(
            "Only use scenes that fully cover each site's drawn region (recommended)",
            value=True, key="batch_full_coverage",
        )
        st.caption(
            f"Uses the cloud-edge buffer and temporal-anomaly settings from tab 2's Advanced "
            f"sections for every site in this batch (currently: Sentinel-2 buffer "
            f"{st.session_state.cloud_buffer_s2}px, Landsat buffer {st.session_state.cloud_buffer_ls}px, "
            f"temporal anomaly {'enabled' if st.session_state.enable_temporal_anomaly else 'disabled'}) "
            "- adjust those in tab 2 first if needed. Each site's own scenes are still fetched and "
            "processed in memory-bounded batches (see tab 2's memory notes), one site at a time - "
            "nothing from any site is ever written to the server's disk."
        )

        if st.button("Run batch analysis", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def batch_progress_cb(done, total, message):
                status_text.write(message)
                if total:
                    progress_bar.progress(min(done / total, 1.0))

            with st.spinner(f"Running {len(st.session_state.site_queue)} site(s)..."):
                combined_df, per_site_dfs, per_site_errors = fetch.run_batch_analysis(
                    sites=st.session_state.site_queue,
                    start_date=b_start.isoformat(),
                    end_date=b_end.isoformat(),
                    max_cloud=b_max_cloud,
                    products_json_path=PRODUCTS_JSON,
                    min_roi_coverage=fetch.FULL_COVERAGE_THRESHOLD if b_require_full_coverage else 0,
                    progress_cb=batch_progress_cb,
                    cloud_buffer_px={
                        "sentinel2": st.session_state.cloud_buffer_s2, "landsat": st.session_state.cloud_buffer_ls,
                    },
                    enable_temporal_anomaly=st.session_state.enable_temporal_anomaly,
                    temporal_anomaly_threshold=st.session_state.temporal_anomaly_threshold,
                    temporal_anomaly_percentile=st.session_state.temporal_anomaly_percentile,
                    temporal_anomaly_min_obs=st.session_state.temporal_anomaly_min_obs,
                )
                st.session_state.batch_combined_df = combined_df
                st.session_state.batch_per_site_dfs = per_site_dfs
                st.session_state.batch_per_site_errors = per_site_errors
                status_text.write(f"Done - {len(combined_df)} scenes across {len(per_site_dfs)} site(s).")

        if st.session_state.batch_per_site_errors:
            st.warning(
                "Some sites failed and were skipped:\n\n"
                + "\n".join(f"- **{name}**: {err}" for name, err in st.session_state.batch_per_site_errors.items())
            )

        batch_combined = st.session_state.batch_combined_df
        if batch_combined is not None and len(batch_combined) > 0:
            st.divider()
            st.subheader("Per-site summary")
            summary_rows = []
            for name, site_df in (st.session_state.batch_per_site_dfs or {}).items():
                if site_df is None or len(site_df) == 0:
                    summary_rows.append(dict(
                        site=name, n_scenes=0, n_open=0, n_closed=0, n_indeterminate=0, mean_monthly_pct_closed=None,
                    ))
                    continue
                counts = aggregate.summary_counts(site_df)
                prop = aggregate.mean_monthly_proportion_closed(site_df)
                summary_rows.append(dict(
                    site=name, n_scenes=counts["n_total"], n_open=counts["n_open"], n_closed=counts["n_closed"],
                    n_indeterminate=counts["n_indeterminate"],
                    mean_monthly_pct_closed=round(prop * 100, 1) if prop is not None else None,
                ))
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            dl1, dl2, _ = st.columns([1, 1, 2])
            with dl1:
                st.download_button(
                    "Download combined CSV (all sites)",
                    data=batch_combined.to_csv(index=False).encode("utf-8"),
                    file_name="batch_results_combined.csv",
                    mime="text/csv",
                )
            with dl2:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, site_df in (st.session_state.batch_per_site_dfs or {}).items():
                        if site_df is None or len(site_df) == 0:
                            continue
                        csv_bytes = site_df.to_csv(index=False).encode("utf-8")
                        zf.writestr(f"{name}_results.csv", csv_bytes)
                st.download_button(
                    "Download per-site CSVs (zip)",
                    data=zip_buf.getvalue(),
                    file_name="batch_results_per_site.zip",
                    mime="application/zip",
                )


with tab_upload:
    st.subheader("Analyse previously exported results")
    st.write(
        "Upload a results CSV this app previously exported (tab 2's 'Download results as CSV', "
        "tab 3's combined or per-site downloads, or an older session) to see the same summary "
        "statistics and time-series plot again without re-fetching anything from DEA."
    )
    uploaded = st.file_uploader("Results CSV", type=["csv"], key="upload_csv")
    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            required_cols = {"date", "sensor", "status"}
            missing = required_cols - set(up_df.columns)
            if missing:
                st.error(
                    f"This CSV is missing required column(s): {', '.join(sorted(missing))}. "
                    "Expected the same columns this app's own results CSV export has."
                )
            else:
                up_df["date"] = pd.to_datetime(up_df["date"])
                # Dedupe per site (not globally) if this is a multi-site combined
                # CSV - a naive global dedupe-by-date would otherwise wrongly drop
                # rows just because two different sites happened to share a date.
                if "site" in up_df.columns and up_df["site"].nunique() > 1:
                    up_df = up_df.groupby("site", group_keys=False).apply(aggregate.prefer_sentinel_on_shared_dates)
                else:
                    up_df = aggregate.prefer_sentinel_on_shared_dates(up_df)
                st.session_state.uploaded_results_df = up_df
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")

    up_df = st.session_state.uploaded_results_df
    if up_df is not None and len(up_df) > 0:
        st.divider()
        render_results_summary(up_df, key_prefix="upload", download_filename="uploaded_results_reexport.csv")
        st.caption(
            "Scene preview isn't available here since this data wasn't just fetched from DEA - "
            "load the matching site in tab 1 and use tab 2's scene preview if you want to "
            "inspect a specific date's imagery."
        )
        with st.expander("Full results table"):
            st.dataframe(up_df)
