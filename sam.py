import torch
from PIL import Image
import requests
import numpy as np
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

img_url = "raw_image.png"
raw_image = Image.open(img_url).convert("RGB")
input_points = [[[1323, 832]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores

# masks: list length B; take first image → (num_points, num_masks, H, W)
masks_tensor = masks[0]
# scores: (B, num_points, num_masks); take first image and point
scores_tensor = scores[0, 0].detach().cpu()
best_mask_idx = torch.argmax(scores_tensor).item()

best_mask = masks_tensor[0, best_mask_idx]  # (H, W)
best_mask_bool = (best_mask > 0.5).numpy()

# Save binary mask with original size: object white (255), background black (0)
object_mask_uint8 = (best_mask_bool.astype(np.uint8) * 255)
Image.fromarray(object_mask_uint8, mode="L").save("object_mask.png")

orig_np = np.array(raw_image)
image_np = orig_np.copy()
image_np[best_mask_bool] = 0
Image.fromarray(image_np).save("object_removed.png")

alpha = (best_mask_bool * 255).astype(np.uint8)
rgba = np.dstack([orig_np, alpha])
Image.fromarray(rgba, mode="RGBA").save("object_only.png")

object_rgb = orig_np.copy()
object_rgb[~best_mask_bool] = 0
ys, xs = np.where(best_mask_bool)
if ys.size > 0 and xs.size > 0:
    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1
    cropped_object = object_rgb[y_min:y_max, x_min:x_max]
    Image.fromarray(cropped_object).save("object_cropped.png")
else:
    Image.fromarray(object_rgb).save("object_cropped.png")