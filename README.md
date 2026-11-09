# Pack Oriented Rectangles into Polygons (QGIS 3 Processing Algorithm)

[![QGIS 3](https://img.shields.io/badge/QGIS-3.x-589632.svg?logo=qgis&logoColor=white)](https://qgis.org/)
[![Python 3](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

A native **QGIS 3 Processing Algorithm** that solves the **2D Constrained Polygon Packing / Nesting Problem** for spatial datasets.

It automatically finds the natural orientation of each polygon, evaluates candidate rotation angles and sub-cell phase shifts, and packs the **maximum possible number of fixed-size rectangular modules** (e.g., building footprints, parking stalls, solar arrays, modular units) that fit **100% completely inside** each polygon feature.

---

## The Problem and Why Standard QGIS Tools Fall Short

Standard GIS tools such as `Create Grid` or `Fishnet` generate fixed, static, axis-aligned grids across a bounding box. When applied to real-world cadastral parcels, urban infill sites, or irregular boundaries:
- Standard grids ignore the parcel's natural street line or principal orientation.
- Even if a single orientation angle is provided, a static grid alignment frequently clips boundary corners, resulting in sub-optimal or zero fitted rectangles.
- Users cannot easily test multiple orientations and phase offsets to find the true mathematical maximum capacity per parcel.

### The Solution:
`pack_oriented_rectangles.py` runs an **adaptive optimization loop per feature**:
1. Computes the **Minimum Rotated Bounding Rectangle (MRR)** to discover the parcel's principal orientation axis.
2. Explores multiple candidate rotations: principal axis, transverse (90 degrees), coordinate axis (0 degrees), and diagonal (45 degrees).
3. Evaluates spatial sub-grid phase offsets (e.g., shifting by half-width / half-length) to avoid corner clipping.
4. Uses strict interior containment (`contains`) to guarantee zero parcel boundary overhang.
5. Selects the configuration that yields the **global maximum module count** (max N) for each individual feature.

```
       Standard Axis-Aligned Grid                     Adaptive Oriented Packing (This Tool)
   +--------------------------------+            +--------------------------------+
   |         /-----------/          |            |         /-----------/          |
   |        /  [X]   [X]/           |            |        / [#] [#] [#]/          |
   |       /   [X]     /            |   ===>     |       / [#] [#] [#]/           |
   |      /___________/             |            |      /___________/             |
   |   (clips boundaries = 0 fit)   |            |   (aligned to axis = 6 fit)    |
   +--------------------------------+            +--------------------------------+
```

---

## Key Features and Performance

- **Fast and Vectorized:** Built using NumPy meshgrids and vectorized Shapely C++ geometry operations. Processes thousands of complex parcels in seconds.
- **Sliver and Narrow Polygon Filter:** Automatically discards narrow corridors and sliver geometries using negative buffer erosion (inscribed circle test) before grid generation, saving compute time.
- **Zero Dependencies:** Uses only standard QGIS 3 core libraries (`qgis.core`, `qgis.PyQt`, `numpy`, `shapely`) already bundled with every QGIS installation.
- **Dual Output Layers:**
  1. **Evaluated Polygons Layer:** Enriches original polygons with capacity metrics (`module_count`, `packed_area_m2`, `coverage_pct`, `capacity_category`).
  2. **Packed Rectangles Layer:** Generates individual, properly oriented polygon features for every placed rectangular module.

---

## Practical Applications

- **Urban Planning and Infill Capacity:** Testing housing capacity on vacant urban lots and brownfields using standard house footprints (e.g. 6 x 10 m, 16 x 20 m).
- **Solar Farm and Agrivoltaic Siting:** Inscribing standard PV table/tracker layouts within irregular field boundaries.
- **Parking Lot Layouts:** Estimating parking bay capacity in complex parcel boundaries.
- **Modular Housing and Tiny Home Parks:** Rapid spatial feasibility checks for container homes or prefab structures.
- **Storage and Logistics:** Space utilization analysis in industrial yards and warehouses.

---

## Installation

No compilation or external pip dependencies required. Works on Windows, macOS, and Linux.

### Method 1: QGIS GUI (Recommended)
1. Download [`pack_oriented_rectangles.py`](pack_oriented_rectangles.py) from this repository.
2. In QGIS, open the **Processing Toolbox** (`Ctrl + Alt + T`).
3. Click the **Python icon** in the Processing Toolbox toolbar and select **Add Script to Toolbox...**
4. Select the downloaded `pack_oriented_rectangles.py` file.
5. The algorithm will immediately appear under **Scripts > Vector analysis > Pack oriented rectangles into polygons**.

### Method 2: Copy to QGIS User Profile Directory
Copy `pack_oriented_rectangles.py` directly into your QGIS scripts directory:
- **Windows:** `%APPDATA%\QGIS\QGIS3\profiles\default\processing\scripts\`
- **Linux:** `~/.local/share/QGIS/QGIS3/profiles/default/processing/scripts/`
- **macOS:** `~/Library/Application Support/QGIS/QGIS3/profiles/default/processing/scripts/`

Restart QGIS or right-click the Processing Toolbox and select **Refresh**.

---

## Parameters and Settings

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`Input polygon layer`** | Vector (Polygon) | *Required* | Layer containing parcels, lots, or zones to be evaluated. Must use a projected CRS (meters/feet). |
| **`Rectangle width [m]`** | Distance (Float) | `6.0` | Short side / width of the rectangular module in map CRS units. |
| **`Rectangle length [m]`** | Distance (Float) | `10.0` | Long side / length of the rectangular module in map CRS units. |
| **`Evaluated polygons`** | Vector Output | `[Create temporary layer]` | Output polygon layer containing original features + capacity attributes. |
| **`Packed rectangles`** | Vector Output | `[Create temporary layer]` | Output polygon layer with each individual placed module geometry. |

> [!NOTE]
> Make sure your input layer is in a **Projected Coordinate Reference System (CRS)** with metric units (e.g. UTM, State Plane, EPSG:5514, EPSG:3857) rather than degrees (EPSG:4326), so that width and length correspond to real-world ground distances.

---

## Output Schema

### 1. Evaluated Polygons Layer (`Evaluated polygons`)
Contains all original attributes from the input layer plus:
- `fits_module` *(Integer)*: `1` if at least one module fits completely inside; `0` otherwise.
- `module_count` *(Integer)*: Maximum number of modules successfully packed.
- `packed_area_m2` *(Double)*: Total footprint area occupied by placed modules (m2).
- `coverage_pct` *(Double)*: Percentage of total parcel area occupied by placed modules (%).
- `capacity_category` *(String)*: Qualitative classification:
  - `Single module (1 rectangle)`
  - `Small capacity (2 - 5 rectangles)`
  - `Medium capacity (6 - 20 rectangles)`
  - `Large capacity (> 20 rectangles)`
  - `Narrow corridor (width < X m)`
  - `Insufficient area (< X m2)`

### 2. Packed Rectangles Layer (`Packed rectangles`)
Individual polygon geometries of each fitted module:
- `rect_id` *(Integer)*: Unique sequential ID.
- `parent_id` *(Integer)*: Feature ID of the parent polygon.
- `width_m` *(Double)*: Module width.
- `length_m` *(Double)*: Module length.
- `area_m2` *(Double)*: Module surface area.

---

## Sample Data and Quick Test

A sample test dataset is provided in [`sample_data/sample_parcels.geojson`](sample_data/sample_parcels.geojson) (Projected in EPSG:3857).

1. Drag and drop `sample_parcels.geojson` into QGIS.
2. Run **Pack oriented rectangles into polygons** with width = `6` and length = `10`.
3. View the fitted modules immediately aligned to each parcel's unique orientation!

---

## License

This project is licensed under the [MIT License](LICENSE) - feel free to use, modify, and distribute it in commercial, academic, or open-source projects.
