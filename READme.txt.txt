# Automated Mapping Engine
High-Resolution Asset Classification via Deep Learning & Contextual Geometry.

## 1. Required Installations & Extensions
You must have Python 3.10+ installed. Open your terminal inside this project folder and run the following command to install all required dependencies:

pip install -r requirements.txt

*(Windows Note: If `rasterio` throws a GDAL build error during pip install, run `conda install -c conda-forge rasterio` instead).*

## 2. How to Launch
1. Double-click the **`START_SYSTEM.bat`** file. This automatically bypasses Streamlit's default memory limits and allocates 4GB for the drone images.
2. When the browser opens, drag and drop your `.tif` file into the upload zone and hit **Initiate Scan**.
3. View the interactive web viewer to filter prediction classes, or click the generated folder link to drag the final map directly into QGIS.

## 3. Known Constraints
* **QGIS File Lock:** If you have a generated map currently open in QGIS, the app cannot overwrite it. Remove the layer in QGIS before scanning the same village twice.


============================================================================
GeoAI - ENGINEERING HURDLES & SYSTEM TACKLES
=============================================================================
Documenting the technical challenges faced during pipeline development 
and the engineering solutions implemented for stable local deployment.

-----------------------------------------------------------------------------
1. THE 2GB MEMORY BOTTLENECK & STREAMLIT UPLOAD LIMITS
-----------------------------------------------------------------------------
* The Hurdle: Web browsers and Streamlit natively crash when attempting to hold a 2GB+ drone orthomosaic in temporary memory. Streamlit enforces a hard 200MB limit, and bypassing it often leads to Chrome/Edge Out-Of-Memory (OOM) tab crashes.
* The Tackle: We decoupled the UI from the heavy lifting. Instead of holding the file in RAM, the app instantly buffers the uploaded file to a local `Input_Uploads` directory. We also created a 1-click `START_SYSTEM.bat` launcher that forces Streamlit's `--server.maxUploadSize 4000` configuration, ensuring stable 4GB file handling without browser freezing.

-----------------------------------------------------------------------------
2. "GHOST REFRESHES" & UI STATE LOSS
-----------------------------------------------------------------------------
* The Hurdle: Streamlit's architecture reruns the entire Python script from top to bottom every time a user interacts with a widget. When implementing our Interactive Map Inspector, selecting a class from the dropdown would cause the app to "forget" the Deep Learning scan had finished, wiping the screen blank.
* The Tackle: We implemented persistent memory using `st.session_state`. By storing boolean flags (`scan_complete`) and file paths in the session state, the app bypasses the 3-minute inference loop on UI refreshes and safely re-renders only the Map Inspector.

-----------------------------------------------------------------------------
3. BROWSER CRASHES ON HIGH-RES RENDERING
-----------------------------------------------------------------------------
* The Hurdle: Attempting to render a 2GB, multi-band `.tif` file directly in a web dashboard causes an immediate GPU crash on the client side.
* The Tackle: We built a "Dynamic Downsampler." Instead of loading the full map, the UI uses `rasterio` to crack open the final map and selectively read it into a web-safe 1024x1024 NumPy array using `Resampling.nearest`. This preserves strict class boundaries (preventing color bleeding) while keeping the UI buttery smooth.

-----------------------------------------------------------------------------
4. WINDOWS SUBPROCESS & UNICODE DECODE ERRORS
-----------------------------------------------------------------------------
* The Hurdle: Running our PyTorch engine (`run_inference.py`) and OpenCV engine (`geospatial_ai.py`) as background subprocesses caused silent crashes. Windows attempts to decode Python's terminal output using the outdated `cp1252` charmap, which instantly crashes when encountering progress bar characters or emojis.
* The Tackle: We strictly enforced `encoding="utf-8"` in all `subprocess.run()` calls and toggled `capture_output=False`. This safely routes the heavy inference logs directly to the user's terminal window instead of crashing the Streamlit wrapper.

-----------------------------------------------------------------------------
5. THE QGIS "FILE LOCK" DENIAL OF SERVICE
-----------------------------------------------------------------------------
* The Hurdle: In geospatial workflows, evaluators frequently open generated maps in QGIS. If a user runs a second AI scan on the same village while the previous map is still open in QGIS, the Python script throws a fatal `Permission denied` error because QGIS locks the file to prevent corruption.
* The Tackle: We built dynamic file naming (`f"{village_name}_Final_Map.tif"`). We also explicitly documented this behavior in the UI and README, advising users to remove the QGIS layer before rescanning, or to rename their input files to generate versioned maps (e.g., `_V2.tif`).

-----------------------------------------------------------------------------
6. TILING ARTIFACTS & "MINECRAFT" ROADS
-----------------------------------------------------------------------------
* The Hurdle: Because the U-Net model processes the massive image in 512x512 chunks, the resulting geometric predictions for roads and lakes often looked blocky or pixelated at the stitch seams.
* The Tackle: We engineered a heavy Morphological OpenCV Engine. Instead of standard square kernels, we forced `cv2.getStructuringElement(cv2.MORPH_ELLIPSE)`. By applying iterative Dilation and Closure with elliptical kernels, we smoothed the blocky artifacts into natural, curving roads and rounded water bodies.

-----------------------------------------------------------------------------
7. FALSE POSITIVES IN DIRT COURTYARDS (THE TEXTURE VETO)
-----------------------------------------------------------------------------
* The Hurdle: The model frequently confused dry, dusty dirt inside building courtyards for unpaved roads due to identical HSV color profiles.
* The Tackle: We developed the "Texture Veto" system. By generating a Canny Edge mask and applying aggressive dilation, we mapped areas of high texture (grass/dirt/noise) and used it to explicitly veto flat areas (roads/water) from being misclassified. Furthermore, building footprint masks were inverted and used as a secondary veto to ensure roads could never spawn inside a house.
=============================================================================