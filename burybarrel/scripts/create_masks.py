from pathlib import Path

import cv2
import dill as pickle
from lang_sam import LangSAM
import numpy as np
from PIL import Image
from tqdm import tqdm

from burybarrel.langsam_utils import display_image_with_masks

def run(imgdir, text_prompt, outdir, box_threshold=0.3, text_threshold=0.25, closekernelsize: int=0):
    imgdir = Path(imgdir)
    imgpaths = sorted(list(imgdir.glob("*.png")) + list(imgdir.glob("*.jpg")))
    outdir = Path(outdir)
    maskcomp_dir = outdir / "maskcomp"
    maskcomp_dir.mkdir(parents=True, exist_ok=True)
    maskdebug_dir = outdir / "maskdebug"
    maskdebug_dir.mkdir(parents=True, exist_ok=True)
    langsam_model = LangSAM()

    bboxes = []
    for i, imgpath in enumerate(tqdm(imgpaths)):
        image_pil = Image.open(imgpath).convert("RGB")

        results = langsam_model.predict(
            [image_pil], [text_prompt], box_threshold=box_threshold, text_threshold=text_threshold
        )[0]
        boxes = results["boxes"]
        masks = results["masks"]
        logits = results["scores"]

        if len(masks) == 0:
            print(f"No objects of the '{text_prompt}' prompt detected in the image.")
        else:
            masks_np = [mask for mask in masks]
            if closekernelsize > 0:
                # kernel = np.ones((closekernelsize, closekernelsize))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closekernelsize, closekernelsize))
                masks_np = [cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel) for mask_np in masks_np]

            bbox_mask_path = maskcomp_dir / f"{imgpath.stem}_img_with_mask.png"
            bbox_mask_path.parent.mkdir(parents=True, exist_ok=True)
            display_image_with_masks(image_pil, masks_np, boxes, logits, figwidth=13, savefig=bbox_mask_path, all_masks=True, show=False, show_confidence=True)
            
            # jank workaround for excluding those masks that are just supersets
            # of the barrel itself
            boxareas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            # minarea_idx = np.argmin(boxareas)
            # as it turns out jellyfish are detected for some reason, and they are smaller
            # welp, just take the first one, it's usually the barrel anyway right?
            minarea_idx = 0

            # save masks
            for i, mask_np in enumerate(masks_np):
                # each box is x_min, y_min, x_max, y_max
                bbox = boxes[i]
                mask_path = maskdebug_dir / f"{imgpath.stem}_mask_{i+1}.png"
                mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_image.save(mask_path)
            
            bbox = boxes[minarea_idx]
            mask_np = masks_np[minarea_idx]
            mask_path = outdir / f"{imgpath.stem}.png"
            mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_image.save(mask_path)
            bboxes.append(bbox)

    bboxes = np.array(bboxes, dtype=int)
    with open(outdir / "bboxes.pickle", "wb") as f:
        pickle.dump(bboxes, f)
