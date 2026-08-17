# Estuary Mouth Monitor (cloud-hosted build)

This is the cloud-hosted, run-in-the-browser build of the Estuary Mouth
Monitor app, deployed on **Streamlit Community Cloud**
(share.streamlit.io). It's the counterpart to the local desktop version at
[Estuary-opening-local-app](https://github.com/michaelshackleton/Estuary-opening-local-app) -
same map, same open/closed/indeterminate classification, same DEA data
source. The differences below are only the bits that don't make sense on a
shared server with no display or persistent filesystem of its own.

## What's different from the desktop version

**Site-layer save/load is a file download/upload, not a folder picker.**
The desktop version opens a native Windows "browse for folder" dialog and
writes Esri Shapefiles to it. A server has no access to your computer's
filesystem and no display to pop a dialog on, so this version instead:

- **Save**: click "Download site layers (.json)" in the sidebar - your
  browser downloads one JSON file containing the ROI, inside/outside lines,
  and any structures.
- **Load**: use the "Upload a previously downloaded site-layers .json file"
  box in the sidebar, then click "Load uploaded site layers".

Keep that JSON file somewhere sensible (a folder per site works well) and
you can restore a site instantly without redrawing.

**No raster ever touches the server's disk.** The desktop version caches
fetched scenes to a `data_cache/` folder on your own machine, which is
fine there - it's your machine. On a shared host like Streamlit Community
Cloud, every visitor's browser talks to the *same* running server process.
`st.session_state` is isolated per browser session, but a file written to
the server's actual filesystem is not - a disk cache keyed only by site
name + sensor + date would let two different people who happen to type the
same site name (e.g. both leave it at the default "new_site") and preview
an overlapping date silently load *each other's* cached raster, clipped to
a *different* ROI. To avoid that cross-user risk on a multi-tenant host,
this build never writes a scene to disk at all. The scene preview keeps
only the single most-recently-viewed scene in memory
(`st.session_state.preview_scene`), automatically replaced (and the old
one dropped) the moment you click a different point on the results plot,
and automatically gone the moment your browser session ends - nothing to
clear, nothing shared between users. If you want a local copy of a scene,
use the "Download this scene as GeoTIFF" button under the preview - it
streams straight to your own computer, never the server. There's no
"cache every scene during this run" option either, for the same reason -
the full-run analysis already only ever holds one scene in memory at a
time to compute its connectivity result, then discards it.

**Long analysis runs are processed in batches, not all at once.**
The desktop version's `run_site_analysis()` loads every matched scene's
full band set for the whole requested date range into memory in one shot
before processing anything. On a desktop with plenty of RAM that's rarely
an issue; on Streamlit Community Cloud's ~1GB container it can run out of
memory on longer, multi-sensor runs with temporal anomaly detection
enabled. This build fetches and processes each product's scenes in batches
of about six months at a time
(`modules/fetch.py`'s `DEFAULT_BATCH_MONTHS`), discarding each batch before
fetching the next, so peak memory stays roughly flat regardless of how long
a date range you ask for. The one exception is the temporal-anomaly
clear-sky reference, which genuinely needs to see every scene across the
whole range at once to mean what it's supposed to as a percentile - that
part still fetches the full range in one pass, but only 2 bands (blue +
fmask) instead of the full 6-7, which keeps its footprint far smaller than
loading everything would. None of this changes any result - same pixels,
same connectivity calls, same reference values - only how much is held in
memory at once. After each batch, `_release_memory()` forces a garbage
collection pass and asks glibc to actually return freed memory to the OS -
CPython/glibc otherwise tend to hold onto large freed blocks for reuse
rather than releasing them, which in a long run of many batches back to
back can make the process's OS-visible memory keep climbing even though
nothing is really "leaking". If a very large ROI still runs out of memory,
lowering `DEFAULT_BATCH_MONTHS` (currently 6) trades some speed for a
smaller per-batch footprint.

**This only bounds one run's own memory use, not concurrent runs.**
Streamlit Community Cloud runs the whole app as one process shared by
every visitor - each browser session gets isolated `st.session_state`, but
they all draw from the same memory ceiling. If multiple people run an
analysis at close to the same time, their peak memory needs stack on top
of each other. There is currently no mechanism to queue or limit
overlapping analysis runs.

**`opencv-python-headless` instead of `opencv-python`.** The vendored
`vendor/rsderiv/sen1.py` module (a Sentinel-1 SAR speckle filter) imports
`cv2` at module load time, even though this app never processes SAR data
and never calls that module's functions - `vendor/rs_processing.py`
unconditionally imports it as a side effect. Regular `opencv-python` needs
a system graphics library (`libGL.so.1`) that isn't present on Streamlit
Cloud's minimal Linux image and will fail to import; the headless build
provides the same `cv2` API without that dependency.

**Nothing else changed.** The map, the drawing tools, the connectivity
algorithm (`modules/connectivity.py`), the DEA fetch pipeline
(`modules/fetch.py`, `vendor/`), the results aggregation, the scene preview
and its diagnostic overlays are all identical to the desktop app.

## Deploying to Streamlit Community Cloud

1. Push this `Estuary openings app/` folder to a GitHub repo (it's fully
   self-contained - nothing outside this folder is needed).
2. On [share.streamlit.io](https://share.streamlit.io), click "New app",
   point it at that repo, and set the main file path to `app.py`. If this
   folder isn't the repo root, set the app's "Main file path" to
   `Estuary openings app/app.py` and, in "Advanced settings", the working
   directory the same way (or just make this folder its own repo).
3. No secrets or API keys are needed - Digital Earth Australia's STAC
   endpoint (`https://explorer.sandbox.dea.ga.gov.au/stac`), which is all
   this app talks to, is public and requires no authentication.
4. `runtime.txt` pins Python 3.11, which has readily-available prebuilt
   wheels for the geospatial dependencies (rasterio, GDAL-backed
   geopandas, etc.) - if Streamlit Cloud's build fails on a dependency,
   that's the first thing worth checking (a `packages.txt` with apt-level
   GDAL packages might be needed on some images, though modern rasterio/
   geopandas wheels usually bundle everything they need).

## Running locally instead

Same as any Streamlit app:

```
cd "Estuary openings app"
python -m venv .venv
.venv\Scripts\activate   # or source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Folder layout

- `app.py` - the Streamlit UI (map + draw tools + run controls + results).
- `modules/connectivity.py` - the NDWI/oa_fmask water-masking and
  path-connectivity logic. No Streamlit or STAC code in here.
- `modules/region.py` - manages the drawn ROI/lines/structures layers,
  including the JSON-bundle save/load described above.
- `modules/fetch.py` - queries DEA's STAC catalogue and loads scenes for
  the drawn ROI, via the vendored `vendor/rs_data.py` /
  `vendor/rs_processing.py`.
- `modules/aggregate.py` - builds the results table and the mean-monthly
  proportion-closed statistic.
- `vendor/` - self-contained copies of the DEA-fetch helpers this app
  depends on, so it doesn't need anything else alongside it to run.
- `config/products.json` - the two DEA product definitions used
  (`landsat_full`, `sentinel_full`).

There is no `data_cache/` folder in this build - see "No raster ever
touches the server's disk" above.

See the [local desktop app's README](https://github.com/michaelshackleton/Estuary-opening-local-app)
for the full design-decisions writeup - open/closed combination rule,
cloud-edge buffer, temporal anomaly detection, connectivity diagnostic,
etc. - all of that applies unchanged here.
