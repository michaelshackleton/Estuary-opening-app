"""
Manages the user-drawn layers for a site (ROI polygon, inside/outside
lines, optional structure polygons, optional sandbar polygons), including
saving/loading them as a single GeoJSON-based bundle - so an analysis can
be reproduced later without redrawing.

CLOUD-HOSTING NOTE: this module is a fork of Claude_script/modules/region.py
(the original desktop app's version), adapted for Streamlit Community Cloud
hosting. The desktop version saves/loads Esri Shapefiles to/from a folder
the user picks via a native OS folder-browse dialog - neither of those
(multi-file shapefiles, native dialogs) works on a headless server the
user's browser has no filesystem access to. This version instead bundles
roi/lines/structures into one JSON file (to_json_bytes/from_json_bytes),
meant to be handed to Streamlit's st.download_button (save) and
st.file_uploader (load) - see app.py's sidebar. This also means the app no
longer depends on a shapefile driver (fiona/pyogrio) for site-layer I/O at
all, which is one less thing that can go wrong in a minimal cloud
container.

Layers are stored in EPSG:4326 (lat/lon, what the Leaflet draw tools return)
and reprojected to a local UTM zone on demand for analysis - the UTM zone is
auto-detected from the ROI centroid so this works anywhere in Australia
without the user having to know their MGA zone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
from shapely.geometry import shape

WGS84 = "EPSG:4326"


def auto_utm_epsg(lon: float, lat: float) -> int:
    """WGS84 UTM zone EPSG code for a given lon/lat. Australia is entirely
    southern hemisphere, but this works globally for robustness."""
    zone = int((lon + 180) // 6) + 1
    return (32700 if lat < 0 else 32600) + zone


def name_from_centroid(roi_gdf: gpd.GeoDataFrame, prefix: str = "site", decimals: int = 4) -> str:
    """Builds a site name like 'site_38.1234S_140.5678E' from the ROI
    polygon's centroid (computed in WGS84 regardless of the GeoDataFrame's
    current CRS) - lets sites be named automatically and consistently
    rather than requiring the user to type a unique name for every site,
    which matters once you're drawing many sites in one session for a
    batch run. N/S and E/W suffixes are used instead of a signed number so
    the name reads naturally as a coordinate and stays safe as a filename
    if downloaded. Two sites would only collide if their ROI centroids
    matched to `decimals` places (4 decimals is ~11m at the equator) -
    vanishingly unlikely for genuinely different estuaries."""
    centroid = roi_gdf.to_crs(WGS84).geometry.iloc[0].centroid
    lat, lon = centroid.y, centroid.x
    ns = "S" if lat < 0 else "N"
    ew = "W" if lon < 0 else "E"
    return f"{prefix}_{abs(lat):.{decimals}f}{ns}_{abs(lon):.{decimals}f}{ew}"


@dataclass
class SiteLayers:
    name: str
    roi: gpd.GeoDataFrame  # single polygon, EPSG:4326
    lines: gpd.GeoDataFrame  # two lines with a 'position' column: inside/outside, EPSG:4326
    structures: Optional[gpd.GeoDataFrame] = None  # optional polygons, EPSG:4326
    # Optional polygons marking sandbars - areas that can genuinely
    # alternate between exposed sand and open water. Unlike `structures`
    # (always forced to water), a sandbar is NOT forced either way - it's
    # exempted only from the temporal-anomaly check (see
    # connectivity.process_scene's `sandbars_gdf` param), so its pixels
    # keep whatever the ordinary NDWI/fmask classification found, rather
    # than being flagged as no-data/uncertain just because a sandbar
    # popping up (or a wet one going under) makes the pixel read
    # differently from its own clear-sky history.
    sandbars: Optional[gpd.GeoDataFrame] = None  # optional polygons, EPSG:4326

    # -- validation ---------------------------------------------------
    def validate(self) -> list[str]:
        """Returns a list of human-readable problems, empty if valid."""
        problems = []
        if self.roi is None or len(self.roi) == 0:
            problems.append("No region-of-interest polygon has been drawn yet.")
        elif len(self.roi) > 1:
            problems.append(
                f"Region of interest has {len(self.roi)} polygons - only one is expected. "
                "Using the first one."
            )
        if self.lines is None or len(self.lines) == 0:
            problems.append("No inside/outside lines have been drawn yet.")
        else:
            if "position" not in self.lines.columns:
                problems.append("Lines layer is missing a 'position' column.")
            else:
                positions = self.lines["position"].tolist()
                if positions.count("inside") != 1:
                    problems.append("Need exactly one line labelled 'inside'.")
                if positions.count("outside") != 1:
                    problems.append("Need exactly one line labelled 'outside'.")
        return problems

    # -- geometry access ------------------------------------------------
    def target_crs(self) -> int:
        """Auto-detected UTM EPSG code from the ROI centroid."""
        centroid = self.roi.to_crs(WGS84).geometry.iloc[0].centroid
        return auto_utm_epsg(centroid.x, centroid.y)

    def inside_line(self, target_crs=None) -> gpd.GeoDataFrame:
        gdf = self.lines[self.lines["position"] == "inside"]
        return gdf.to_crs(target_crs) if target_crs else gdf

    def outside_line(self, target_crs=None) -> gpd.GeoDataFrame:
        gdf = self.lines[self.lines["position"] == "outside"]
        return gdf.to_crs(target_crs) if target_crs else gdf

    def roi_polygon(self, target_crs=None):
        gdf = self.roi.to_crs(target_crs) if target_crs else self.roi
        return gdf.iloc[0].geometry

    def structures_reprojected(self, target_crs) -> Optional[gpd.GeoDataFrame]:
        if self.structures is None or len(self.structures) == 0:
            return None
        return self.structures.to_crs(target_crs)

    def sandbars_reprojected(self, target_crs) -> Optional[gpd.GeoDataFrame]:
        if self.sandbars is None or len(self.sandbars) == 0:
            return None
        return self.sandbars.to_crs(target_crs)

    # -- save / load (GeoJSON bundle, for browser upload/download) --------
    # A single JSON file with embedded GeoJSON FeatureCollections
    # (roi/lines/structures/sandbars) plus the site name - simpler and more
    # portable for a cloud app than a multi-file shapefile bundle, and
    # needs no filesystem access at all (see module docstring). The app's
    # sidebar wires these into st.download_button (save) / st.file_uploader
    # (load).

    def to_geojson_dict(self) -> dict:
        return {
            "name": self.name,
            "roi": json.loads(self.roi.to_crs(WGS84).to_json()),
            "lines": json.loads(self.lines.to_crs(WGS84).to_json()),
            "structures": (
                json.loads(self.structures.to_crs(WGS84).to_json())
                if self.structures is not None and len(self.structures) > 0
                else None
            ),
            "sandbars": (
                json.loads(self.sandbars.to_crs(WGS84).to_json())
                if self.sandbars is not None and len(self.sandbars) > 0
                else None
            ),
        }

    def to_json_bytes(self) -> bytes:
        """Serialises this site's layers to a JSON byte string, ready for
        st.download_button's `data=` argument."""
        return json.dumps(self.to_geojson_dict(), indent=2).encode("utf-8")

    @classmethod
    def from_geojson_dict(cls, d: dict) -> "SiteLayers":
        roi = gpd.GeoDataFrame.from_features(d["roi"]["features"], crs=WGS84)
        lines = gpd.GeoDataFrame.from_features(d["lines"]["features"], crs=WGS84)
        structures = None
        if d.get("structures"):
            structures = gpd.GeoDataFrame.from_features(d["structures"]["features"], crs=WGS84)
        sandbars = None
        if d.get("sandbars"):
            sandbars = gpd.GeoDataFrame.from_features(d["sandbars"]["features"], crs=WGS84)
        return cls(name=d.get("name") or "site", roi=roi, lines=lines, structures=structures, sandbars=sandbars)

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "SiteLayers":
        """Reads a site's layers back from bytes produced by
        to_json_bytes() - the shape returned by st.file_uploader's
        `.getvalue()`. Raises a plain Exception with a readable message on
        malformed input, since this is user-facing (someone uploading the
        wrong file) rather than a programming error."""
        try:
            d = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"That doesn't look like a site-layers JSON file: {e}") from e
        if "roi" not in d or "lines" not in d:
            raise ValueError(
                "That JSON file doesn't have the expected 'roi'/'lines' keys - "
                "is this a site-layers file downloaded from this app?"
            )
        return cls.from_geojson_dict(d)


# --------------------------------------------------------------------------
# Helpers for converting the raw GeoJSON coming back from the Leaflet draw
# control (via streamlit-folium) into GeoDataFrames
# --------------------------------------------------------------------------

def geojson_features_to_gdf(features: list[dict], crs=WGS84) -> gpd.GeoDataFrame:
    """`features` is a list of GeoJSON Feature dicts, as returned in
    st_folium's 'all_drawings' list."""
    geoms = [shape(f["geometry"]) for f in features]
    props = [f.get("properties", {}) or {} for f in features]
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
    return gdf
