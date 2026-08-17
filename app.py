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
from modules.region import SiteLayers, geojson_features_to_gdf  # noqa: E402

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
        site_name="new_site",
        results_df=None,
        raw_records=None,
        run_message="",
        selected_point=None,
        roi_map_v=0,
        lines_map_v=0,
        struct_map_v=0,
        structures_decided=False,
        enable_temporal_anomaly=False,
        temporal_anomaly_threshold=connectivity.DEFAULT_TEMPORAL_ANOMALY_THRESHOLD_DN,
        temporal_anomaly_percentile=10.0,
        temporal_anomaly_min_obs=3,
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


def add_context_layers(m, roi=None, lines=None, structures=None):
    """Adds already-confirmed layers to a map as static (non-editable)
    context so the user can see what they've already drawn while drawing
    the next layer."""
    if roi is not None and len(roi) > 0:
        folium.GeoJson(
            roi, name="ROI", style_function=lambda f: {"color": PASTEL["roi"], "weight": 3, "fillOpacity": 0.08}
        ).add_to(m)
    if lines is not None and len(lines) > 0:
        for _, row in lines.iterrows():
            color = PASTEL["inside"] if row.get("position") == "inside" else PASTEL["outside"]
            folium.GeoJson(row.geometry.__geo_interface__, style_function=lambda f, c=color: {"color": c, "weight": 5}).add_to(m)
    if structures is not None and len(structures) > 0:
        folium.GeoJson(
            structures, name="Structures",
            style_function=lambda f: {"color": PASTEL["structures"], "weight": 2, "fillOpacity": 0.35},
        ).add_to(m)
    return m


def map_center_from_roi():
    if st.session_state.roi_gdf is not None and len(st.session_state.roi_gdf) > 0:
        c = st.session_state.roi_gdf.geometry.iloc[0].centroid
        return (c.y, c.x)
    return AUSTRALIA_CENTER


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
can_save = st.session_state.roi_gdf is not None and st.session_state.lines_gdf is not None
if can_save:
    _site_for_download = SiteLayers(
        name=st.session_state.site_name,
        roi=st.session_state.roi_gdf,
        lines=st.session_state.lines_gdf,
        structures=st.session_state.structures_gdf,
    )
    st.sidebar.download_button(
        "Download site layers (.json)",
        data=_site_for_download.to_json_bytes(),
        file_name=f"{st.session_state.site_name}_site.json",
        mime="application/json",
        help="Saves the current ROI, inside/outside lines and structures to a single file "
             "on your own computer - upload it below any time to restore this exact site "
             "without redrawing.",
    )
else:
    st.sidebar.caption("Draw a region (tab 1) and lines (tab 2) first to enable download.")

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
        st.session_state.structures_decided = True  # loading implies the decision was already made
        st.session_state.site_name = loaded.name
        st.session_state.results_df = None
        st.session_state.clear_sky_references = {}
        st.session_state.preview_scene = None
        st.session_state.roi_map_v += 1
        st.session_state.lines_map_v += 1
        st.session_state.struct_map_v += 1
        st.sidebar.success(f"Loaded '{loaded.name}'.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))

if st.sidebar.button("Start a new (blank) site"):
    st.session_state.roi_gdf = None
    st.session_state.lines_gdf = None
    st.session_state.structures_gdf = None
    st.session_state.structures_decided = False
    st.session_state.results_df = None
    st.session_state.clear_sky_references = {}
    st.session_state.preview_scene = None
    st.session_state.roi_map_v += 1
    st.session_state.lines_map_v += 1
    st.session_state.struct_map_v += 1
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
        passable water.
        """
    )


# --------------------------------------------------------------------------
# Main layout: drawing workflow tabs + run/results tab
# --------------------------------------------------------------------------

tab_home, tab_roi, tab_lines, tab_struct, tab_run = st.tabs(
    ["Home", "1. Region of interest", "2. Inside / outside lines", "3. Structures (optional)", "4. Run & results"]
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

        1. **Region of interest** - draw a box or polygon around the estuary
           mouth on the satellite map. This defines the area imagery is
           fetched and clipped to, so keep it reasonably tight around the
           mouth rather than the whole estuary.
        2. **Inside / outside lines** - draw exactly two lines: one crossing
           the water on the river ("inside") side of the mouth, one crossing
           it on the ocean ("outside") side. On each satellite scene, the app
           checks whether water connects these two lines - if it does, the
           mouth is open on that date.
        3. **Structures (optional)** - if a bridge or causeway crosses the
           estuary near your lines, draw a polygon over it so it's always
           treated as passable water, rather than wrongly breaking the
           connection.
        4. **Run & results** - pick a date range and maximum cloud cover,
           then run the analysis. You'll get a time series plot of
           open/closed/indeterminate status, summary statistics (including
           the mean monthly proportion of time closed), and a scene preview
           where you can click any point on the plot to see exactly what the
           satellite image and classification looked like on that date.

        A few other things worth knowing about, in the sidebar:

        - **Save / load site layers** lets you download your drawn ROI,
          lines and structures as a single file, and re-upload it later to
          pick up an existing site without redrawing.
        - **About the classification** (further down the sidebar) explains
          the open/closed/indeterminate logic in more detail.
        - The "Advanced" sections in tab 4 (cloud-edge buffer, temporal
          anomaly detection) are optional tuning for tricky sites where
          cloud or haze is causing false results - the defaults work well
          for most estuaries, so it's fine to leave them alone starting out.
        """
    )

with tab_roi:
    if st.session_state.roi_gdf is not None:
        st.success("✓ ROI confirmed for this session.")
        view_map = satellite_map(center=map_center_from_roi(), zoom=12)
        add_context_layers(view_map, roi=st.session_state.roi_gdf)
        st_folium(view_map, key=f"map_roi_view_{st.session_state.roi_map_v}", height=600, use_container_width=True)
        if st.button("Redraw ROI"):
            st.session_state.roi_gdf = None
            st.session_state.roi_map_v += 1
            st.rerun()
    else:
        st.write(
            "Use the polygon tool (top-left of the map) to draw a region around the estuary mouth. "
            "This defines both the area rasters are fetched for and clipped to."
        )
        m = satellite_map(center=map_center_from_roi(), zoom=5)
        Draw(
            export=False,
            draw_options={"polygon": True, "polyline": False, "rectangle": True, "circle": False, "marker": False, "circlemarker": False},
            edit_options={"edit": True, "remove": True},
        ).add_to(m)
        map_data = st_folium(m, key=f"map_roi_{st.session_state.roi_map_v}", height=750, use_container_width=True)

        drawings = (map_data or {}).get("all_drawings") or []
        polygons = [f for f in drawings if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]

        if polygons:
            st.info(f"{len(polygons)} polygon(s) drawn. Confirming will use the most recently drawn one.")
            if st.button("Confirm ROI"):
                gdf = geojson_features_to_gdf([polygons[-1]])
                st.session_state.roi_gdf = gdf
                st.rerun()

with tab_lines:
    if st.session_state.roi_gdf is None:
        st.warning("Draw and confirm a region of interest in tab 1 first.")
    elif st.session_state.lines_gdf is not None:
        st.success("✓ Lines confirmed for this session (inside + outside).")
        view_map = satellite_map(center=map_center_from_roi(), zoom=14)
        add_context_layers(view_map, roi=st.session_state.roi_gdf, lines=st.session_state.lines_gdf)
        st_folium(view_map, key=f"map_lines_view_{st.session_state.lines_map_v}", height=600, use_container_width=True)
        if st.button("Redraw lines"):
            st.session_state.lines_gdf = None
            st.session_state.lines_map_v += 1
            st.rerun()
    else:
        st.write(
            "Use the polyline tool to draw exactly two lines: one crossing the river **inside** "
            "the estuary, one crossing the ocean side **outside** the mouth."
        )
        m = satellite_map(center=map_center_from_roi(), zoom=14)
        add_context_layers(m, roi=st.session_state.roi_gdf)
        Draw(
            export=False,
            draw_options={"polygon": False, "polyline": True, "rectangle": False, "circle": False, "marker": False, "circlemarker": False},
            edit_options={"edit": True, "remove": True},
        ).add_to(m)
        map_data = st_folium(m, key=f"map_lines_{st.session_state.lines_map_v}", height=750, use_container_width=True)

        drawings = (map_data or {}).get("all_drawings") or []
        line_feats = [f for f in drawings if f["geometry"]["type"] in ("LineString", "MultiLineString")]

        if line_feats:
            st.write(f"{len(line_feats)} line(s) drawn - assign each one:")
            positions = []
            cols = st.columns(len(line_feats))
            for i, (col, feat) in enumerate(zip(cols, line_feats)):
                with col:
                    pos = st.selectbox(f"Line {i + 1}", ["inside", "outside", "(ignore)"], key=f"line_pos_{i}")
                    positions.append(pos)
            if st.button("Confirm lines"):
                if positions.count("inside") != 1 or positions.count("outside") != 1:
                    st.error("Assign exactly one line as 'inside' and exactly one as 'outside'.")
                else:
                    keep = [(f, p) for f, p in zip(line_feats, positions) if p != "(ignore)"]
                    gdf = geojson_features_to_gdf([f for f, _ in keep])
                    gdf["position"] = [p for _, p in keep]
                    st.session_state.lines_gdf = gdf
                    st.rerun()

with tab_struct:
    if st.session_state.roi_gdf is None:
        st.warning("Draw and confirm a region of interest in tab 1 first.")
    elif st.session_state.structures_decided:
        has_structures = st.session_state.structures_gdf is not None and len(st.session_state.structures_gdf) > 0
        if has_structures:
            st.success(f"✓ {len(st.session_state.structures_gdf)} structure polygon(s) confirmed for this session.")
        else:
            st.success("✓ Confirmed: no structures at this site.")
        view_map = satellite_map(center=map_center_from_roi(), zoom=14)
        add_context_layers(
            view_map, roi=st.session_state.roi_gdf, lines=st.session_state.lines_gdf,
            structures=st.session_state.structures_gdf,
        )
        st_folium(view_map, key=f"map_struct_view_{st.session_state.struct_map_v}", height=600, use_container_width=True)
        if st.button("Redraw structures"):
            st.session_state.structures_gdf = None
            st.session_state.structures_decided = False
            st.session_state.struct_map_v += 1
            st.rerun()
    else:
        st.write(
            "Optional: draw polygons over any structures (bridges, causeways) that cross the "
            "estuary. Pixels under these are always treated as passable water, so a bridge deck "
            "doesn't wrongly break the path between the two lines."
        )
        m = satellite_map(center=map_center_from_roi(), zoom=14)
        add_context_layers(m, roi=st.session_state.roi_gdf, lines=st.session_state.lines_gdf)
        Draw(
            export=False,
            draw_options={"polygon": True, "polyline": False, "rectangle": True, "circle": False, "marker": False, "circlemarker": False},
            edit_options={"edit": True, "remove": True},
        ).add_to(m)
        map_data = st_folium(m, key=f"map_structures_{st.session_state.struct_map_v}", height=750, use_container_width=True)

        drawings = (map_data or {}).get("all_drawings") or []
        struct_polys = [f for f in drawings if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]

        col1, col2 = st.columns(2)
        with col1:
            if struct_polys and st.button("Confirm structures"):
                gdf = geojson_features_to_gdf(struct_polys)
                st.session_state.structures_gdf = gdf
                st.session_state.structures_decided = True
                st.rerun()
        with col2:
            if st.button("No structures at this site"):
                st.session_state.structures_gdf = None
                st.session_state.structures_decided = True
                st.rerun()

with tab_run:
    ready = st.session_state.roi_gdf is not None and st.session_state.lines_gdf is not None
    if not ready:
        st.warning("Complete tabs 1 and 2 (region + lines) before running an analysis.")
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
                cloud_buffer_s2 = st.number_input(
                    "Sentinel-2 buffer (pixels, 10m each)", min_value=0, max_value=100,
                    value=connectivity.DEFAULT_CLOUD_BUFFER_PX["sentinel2"],
                )
            with buf_c2:
                cloud_buffer_ls = st.number_input(
                    "Landsat buffer (pixels, 30m each)", min_value=0, max_value=20,
                    value=connectivity.DEFAULT_CLOUD_BUFFER_PX["landsat"],
                )
        cloud_buffer_px = {"sentinel2": cloud_buffer_s2, "landsat": cloud_buffer_ls}

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

            method_choice = st.radio(
                "Classify using",
                ["Combined (open if either method connects)", "NDWI only", "fmask only"],
                horizontal=True,
                help=(
                    "Combined (the default) calls a scene open if either method finds a "
                    "connected path - this is what feeds the site-level statistics normally. "
                    "NDWI/fmask only shows just that one method's own result, ignoring "
                    "whether the other method agrees - useful for comparing the two or for "
                    "digging into a scene where they disagree (see the connectivity "
                    "diagnostic below for that)."
                ),
            )
            status_col = {
                "Combined (open if either method connects)": "status",
                "NDWI only": "status_ndwi",
                "fmask only": "status_fmask",
            }[method_choice]

            counts = aggregate.summary_counts(combined, status_col=status_col)
            prop_closed = aggregate.mean_monthly_proportion_closed(combined, status_col=status_col)

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
                sub = combined[combined["sensor"] == sensor]
                if len(sub) == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=sub["date"],
                        y=sub[status_col].map(status_y),
                        mode="markers",
                        name=sensor,
                        marker=dict(symbol=symbol, size=10, color=color),
                        # customdata carries (date, sensor) so a click can be matched back
                        # to its row directly - safer than relying on curve/trace index,
                        # since a sensor's whole trace is skipped when it has no scenes.
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
                    data=combined.to_csv(index=False).encode("utf-8"),
                    file_name=f"{st.session_state.site_name}_results.csv",
                    mime="text/csv",
                )

            event = st.plotly_chart(fig, key="ts_plot", on_select="rerun", use_container_width=True)

            selected = None
            if event and event.get("selection", {}).get("points"):
                pt = event["selection"]["points"][0]
                cd = pt.get("customdata")
                if cd:
                    sel_date = pd.Timestamp(cd[0])
                    sel_sensor = cd[1]
                    match = combined[(combined["date"] == sel_date) & (combined["sensor"] == sel_sensor)]
                    if len(match) > 0:
                        selected = match.iloc[0]

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
                    scene_result = connectivity.process_scene(
                        bands, transform, inside_gdf, outside_gdf, structures_gdf, cloud_buffer_px=cloud_buffer_px,
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
