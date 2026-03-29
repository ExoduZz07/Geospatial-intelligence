import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"C:\project_iit\mopr_hybrid_shape_3050.pth"

import os
# ... (keep your existing imports like torch, rasterio, etc.) ...

# 1. Grab the target image passed down from the Streamlit UI
TEST_IMAGE_PATH = os.environ.get("HACKATHON_TARGET_TIF", r"C:\project_iit\data\testing_dataset\BADRA_BARNALA_40044_ORTHO.tif")

# 2. Extract the unique village name
village_name = os.path.splitext(os.path.basename(TEST_IMAGE_PATH))[0]

# 3. Point to the shared "Audit-Proof" output folder
output_dir = r"C:\project_iit\Final_Outputs"
os.makedirs(output_dir, exist_ok=True)

# 4. Set the dynamic output path for the AI Mask
# Note: If your script used a different variable name (like MASK_PATH), just rename it here!
OUTPUT_MASK_PATH = os.path.join(output_dir, f"{village_name}_AI_Mask.tif")

# ... (the rest of your PyTorch U-Net scanning code remains exactly the same, 
# just make sure it saves using the OUTPUT_MASK_PATH variable) ...
PATCH_SIZE = 512

def run_ai_scanner():
    print("[1/2] Waking up the AI Brain...")
    model = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    print(f"\n[2/2] Scanning Test Village (Low-RAM Windowed Mode)...")
    with rasterio.open(TEST_IMAGE_PATH) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": 'uint8', "compress": 'lzw'})

        with rasterio.open(OUTPUT_MASK_PATH, 'w', **meta) as dst:
            for y in tqdm(range(0, src.height, PATCH_SIZE), desc="Scanning Rows"):
                for x in range(0, src.width, PATCH_SIZE):
                    h, w = min(PATCH_SIZE, src.height - y), min(PATCH_SIZE, src.width - x)
                    window = Window(x, y, w, h)
                    
                    img_patch = src.read([1, 2, 3], window=window)
                    img_patch = np.moveaxis(img_patch, 0, -1)
                    
                    # Fix: Pad edges to prevent U-Net dimension crashes
                    pad_h, pad_w = PATCH_SIZE - h, PATCH_SIZE - w
                    if pad_h > 0 or pad_w > 0:
                        img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                        
                    img_tensor = torch.from_numpy(img_patch / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                    
                    with torch.no_grad():
                        pred = model(img_tensor)
                        pred_class = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
                        
                    if pad_h > 0 or pad_w > 0:
                        pred_class = pred_class[:h, :w]
                        
                    dst.write(pred_class, 1, window=window)

    print(f"\n✅ AI Mask saved to: {OUTPUT_MASK_PATH}")

if __name__ == "__main__":
    run_ai_scanner()