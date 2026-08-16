# Estuary Mouth Monitor (cloud-hosted build)

This is a fork of `Claude_script/` (the original desktop Estuary Mouth
Monitor app), adapted to run on **Streamlit Community Cloud**
(share.streamlit.io) instead of on your own machine. Same map, same
open/closed/indeterminate classification, same DEA data source - the only
things that changed are the bits that only make sense on a desktop with its
own display and filesystem.

## What's different from the desktop version

**Site-layer save/load is now a file download/upload, not a folder
picker.** The desktop app opens a native Windows "browse for folder" dialog
and writes Esri Shapefiles to it. A server has no access to your computer's
filesystem and no display to pop a dialog on, so this version instead:

- **Save**: click "Download site layers (.json)" in the sidebar - your
  browser downloads one JSON file containing the ROI, inside/outside lines,
  and any structures.
- **Load**: use the "Upload a previously downloaded site-layers .json file"
  box in the sidebar, then click "Load uploaded site layers".

Keep that JSON file somewhere sensible (a folder per site works well) and
you can restore a site instantly without redrawing.

**The raster cache is ephemeral.** `data_cache/` still speeds up re-clicking
the same scene in the preview during a session, but it lives on the hosted
container's own temporary disk - it's wiped whenever the app restarts or
you push a new deploy. This was already true of the desktop app's cache in
spirit (it's just a convenience, never required for the analysis itself),
but on the desktop it happened to persist between sessions; here it won't.

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
  depends on, so it doesn't need the rest of `rs-utils-main` alongside it.
- `config/products.json` - the two DEA product definitions used
  (`landsat_full`, `sentinel_full`).
- `data_cache/` - ephemeral per-session scene cache (see above).

See `Claude_script/README.md` (the original desktop app) for the full
design-decisions writeup - open/closed combination rule, cloud-edge buffer,
temporal anomaly detection, connectivity diagnostic, etc. - all of that
still applies unchanged here.
