#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import hashlib
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
from samgeo.text_sam import LangSAM
from diffusers import FluxFillPipeline

# -----------------------------
# Utilities
# -----------------------------
def list_images(input_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = []
    for fn in os.listdir(input_dir):
        p = os.path.join(input_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in exts:
            paths.append(p)
    paths.sort()
    return paths


def pil_load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_uint8_mask(mask_bool: np.ndarray) -> Image.Image:
    # FluxFill expects a mask image. White=to edit, Black=keep.
    m = (mask_bool.astype(np.uint8) * 255)
    return Image.fromarray(m, mode="L").convert("RGB")


def sha256_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def make_rng(key: str, nonce: str) -> random.Random:
    seed = sha256_int(f"{key}::{nonce}") % (2**32)
    return random.Random(seed)

# -----------------------------
# Step 1: Semantics (caption) + CLIP scoring
# -----------------------------

def build_clip(device: str):
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    return clip_model, clip_processor


@torch.no_grad()
def clip_text_embeds(clip_model, clip_processor, texts: List[str], device: str) -> torch.Tensor:
    inputs = clip_processor(text=texts, images=None, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def clip_image_embed(clip_model, clip_processor, image: Image.Image, device: str) -> torch.Tensor:
    inputs = clip_processor(text=None, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0)

def detect_present_fids(
    image_paths,
    clip_model,
    clip_processor,
    SUBTLE_FEATURES, 
    OPTIONAL_FIDS,
    device: str,
    present_thr: float = 0.25,
    min_cover: float = 0.5,
):
    feature_map = {f.fid: f for f in SUBTLE_FEATURES}
    fids = list(feature_map.keys())

    # Only perform a CLIP presence test on features that "may be present"; all others are assumed to be present by default.
    detect_fids = [fid for fid in fids if fid in OPTIONAL_FIDS]
    always_fids = [fid for fid in fids if fid not in OPTIONAL_FIDS]

    present_cnt = {fid: 0 for fid in fids}
    mean_sim = {fid: 0.0 for fid in fids}

    # If there are no optional features to detect, simply return with all features present by default.
    if len(detect_fids) == 0:
        cover = {fid: 1.0 for fid in fids}
        mean_sim = {fid: 1.0 for fid in fids}
        present_fids = fids[:]  # 全部
        return present_fids, cover, mean_sim

    # --- Only build the text bank for `detect_fids`. ---
    text_bank, owner = [], []
    for fid in detect_fids:
        for t in feature_map[fid].clip_phrases:
            text_bank.append(t)
            owner.append(fid)

    text_emb = clip_text_embeds(clip_model, clip_processor, text_bank, device)  # [T,D]

    for p in image_paths:
        img = pil_load_rgb(p)
        img_emb = clip_image_embed(clip_model, clip_processor, img, device)  # [D]
        sims = (text_emb @ img_emb.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()

        fid2sim = {fid: -1e9 for fid in detect_fids}
        for s, fid in zip(sims, owner):
            if s > fid2sim[fid]:
                fid2sim[fid] = float(s)

        for fid in detect_fids:
            mean_sim[fid] += fid2sim[fid]
            if fid2sim[fid] >= present_thr:
                present_cnt[fid] += 1

    n = max(1, len(image_paths))
    cover = {fid: present_cnt[fid] / n for fid in fids}
    mean_sim = {fid: mean_sim[fid] / n for fid in fids}

    # `always_fids` is enabled by default: it forces `cover=1` and assigns a fixed `mean_sim` value (used only for printing/logging).
    for fid in always_fids:
        cover[fid] = 1.0
        mean_sim[fid] = 1.0

    # In the `optional` section, elements that satisfy `min_cover` are considered "present"; all elements in the `always` section are included.
    present_optional = [fid for fid in detect_fids if cover[fid] >= min_cover]
    present_fids = list(dict.fromkeys(always_fids + present_optional))

    # Fallback: If none of the optional conditions are met, that's okay (present_fids will still include always_fids).
    return present_fids, cover, mean_sim


# -----------------------------
# Step 3: LangSAM segmentation + union
# -----------------------------
def langsam_select_mask(
    sam: LangSAM,
    image_path: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    topk: int = 1,
    drop_area_ratio: float = 0.40,
    save_prefix: str = None,
) -> Optional[np.ndarray]:
    """
    Policy:
      - Sort masks by logits desc.
      - If topk==1:
          pick first mask in sorted order with area_ratio <= drop_area_ratio;
          if none, pick the smallest-area mask.
      - If topk==2:
          pick top-2 by logits and OR-merge them (ignore area).
    """
    masks, boxes, phrases, logits = sam.predict(
        image_path,
        text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        return_results=True,  # default False in LangSAM docs :contentReference[oaicite:1]{index=1}
    )

    # ---- normalize masks to bool ndarray [N,H,W]
    if isinstance(masks, torch.Tensor):
        m = masks.detach().cpu()
        if m.ndim == 2:
            m = m.unsqueeze(0)
        elif m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0, :, :]
        m = (m > 0.5).numpy()
    elif isinstance(masks, np.ndarray):
        m = masks
        if m.ndim == 2:
            m = m[None, ...]
        elif m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0, :, :]
        m = (m > 0.5)
    else:
        arrs = []
        for x in masks:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            arrs.append(x)
        if len(arrs) == 0:
            return np.zeros((1, 1), dtype=bool)
        m = (np.stack(arrs, axis=0) > 0.5)

    if m.shape[0] == 0:
        return np.zeros((1, 1), dtype=bool)

    # ---- normalize logits to float ndarray [N]
    if isinstance(logits, torch.Tensor):
        l = logits.detach().cpu().float().numpy()
    else:
        l = np.asarray(logits, dtype=np.float32)
    l = l.reshape(-1)

    # Safety: align lengths if mismatch
    if l.shape[0] != m.shape[0]:
        n = min(l.shape[0], m.shape[0])
        l = l[:n]
        m = m[:n]

    idx_sorted = np.argsort(-l)  # logits high -> low

    # save
    if save_prefix is not None:
        for i in idx_sorted.tolist():
            mask_u8 = (m[int(i)].astype(np.uint8) * 255)
            out_name = f"{save_prefix}_{i}.png"
            cv2.imwrite(out_name, mask_u8)

    # ---- compute area ratios
    H, W = m.shape[1], m.shape[2]
    areas = m.reshape(m.shape[0], -1).sum(axis=1).astype(np.float32)
    ratios = areas / float(H * W)

    if int(topk) >= 2:
        # Case 2: top-2 merge, ignore area
        if idx_sorted.size == 1:
            return m[int(idx_sorted[0])].astype(bool)
        sel = idx_sorted[:2]
        return np.any(m[sel], axis=0).astype(bool)

    # Case 1: topk==1, skip huge masks first
    for i in idx_sorted.tolist():
        if ratios[int(i)] <= drop_area_ratio:
            return m[int(i)].astype(bool)

    return None


def patch_select_mask(image: Image.Image, rng: random.Random,
                      patch_ratio: float = 0.16, border_ratio: float = 0.08) -> np.ndarray:
    """
    Returns a boolean mask and a rectangular patch. 
    patch_ratio: the ratio of the patch side length to min(H,W)
    border_ratio: to avoid being too close to the edges
    """
    W, H = image.size
    side = max(8, int(min(W, H) * patch_ratio))
    bx = int(W * border_ratio)
    by = int(H * border_ratio)

    x0_min, x0_max = bx, max(bx, W - side - bx)
    y0_min, y0_max = by, max(by, H - side - by)

    x0 = rng.randint(x0_min, x0_max) if x0_max >= x0_min else 0
    y0 = rng.randint(y0_min, y0_max) if y0_max >= y0_min else 0

    m = np.zeros((H, W), dtype=bool)
    m[y0:y0 + side, x0:x0 + side] = True

    return m


# -----------------------------
# Step 2: prompt generation (feature-related + diverse)
# -----------------------------
def estimate_color_name_from_mask(image: Image.Image, mask: np.ndarray) -> str:
    """
    Roughly estimate the dominant color of the masked region (for inliers such as pupils/hair, use "different color" instead).
    """
    arr = np.array(image).astype(np.uint8)
    m = mask.astype(bool)
    if m.sum() < 20:
        return "dark brown"

    pixels = arr[m]
    # RGB -> HSV
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = float(np.median(hsv[:, 0]))  # 0..179
    s = float(np.median(hsv[:, 1]))  # 0..255
    v = float(np.median(hsv[:, 2]))  # 0..255

    if v < 60:
        return "black"
    if s < 40:
        return "gray"

    if 0 <= h < 10 or 170 <= h <= 179:
        return "red"
    if 10 <= h < 25:
        return "orange"
    if 25 <= h < 40:
        return "yellow"
    if 40 <= h < 85:
        return "green"
    if 85 <= h < 115:
        return "cyan"
    if 115 <= h < 145:
        return "blue"
    if 145 <= h < 170:
        return "purple"
    return "brown"

_PLACEHOLDER_RE = re.compile(r"{(\w+)}")

# Global default slot (used by many features, so don't include it in each FeatureSpec)
GLOBAL_SLOTS = {
    "side": ["left", "right"],
    "shape": ["star", "circle", "crescent"],
}

def build_edit_prompt(
    feature,
    rng: random.Random,
    sample_image: Image.Image,
    sample_mask: Optional[np.ndarray],
) -> str:
    tpl = rng.choice(feature.edit_templates)

    # The slots of the feature override the global defaults.
    slots = dict(GLOBAL_SLOTS)
    slots.update(getattr(feature, "slot_values", {}) or {})
    policies = getattr(feature, "slot_policies", {}) or {}

    fmt = {}
    for name in _PLACEHOLDER_RE.findall(tpl):
        if name in fmt:
            continue

        bank = slots.get(name)
        if not bank:
            # If it's not defined, provide a default value (or raise an error, which is better for helping you discover missing configurations).
            fmt[name] = "blue"
            continue

        # Example: The color should avoid the current main color of the mask (which you previously specified in build_edit_prompt): contentReference[oaicite:5]{index=5}
        if policies.get(name) == "different_from_mask" and sample_mask is not None and sample_mask.sum() > 0:
            cur = estimate_color_name_from_mask(sample_image, sample_mask)
            cand = [x for x in bank if x != cur]
            fmt[name] = rng.choice(cand if cand else bank)
        else:
            fmt[name] = rng.choice(bank)

    return tpl.format(**fmt)

# -----------------------------
# Step 4: Flux Fill edit
# -----------------------------
def build_flux_pipe(model_id: str, device: str, dtype: str):
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    return pipe


@torch.no_grad()
def flux_edit_one(
    pipe: FluxFillPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    seed: int,
    guidance_scale: float,
    steps: int,
    max_sequence_length: int,
) -> Image.Image:
    gen = torch.Generator("cpu").manual_seed(int(seed))
    out = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=image.height,
        width=image.width,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=max_sequence_length,
        generator=gen,
    ).images[0]
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="./data/n000050/set_A")
    ap.add_argument("--output_dir", type=str, default="./data/n000050/set_A_edited")

    ap.add_argument("--domain", type=str, choices=["face", "art_style"], default="face")

    # randomness / security
    ap.add_argument("--key", type=str, default="featmark")
    ap.add_argument("--nonce", type=str, default="0")  # Changing this will ensure that each random result is different.

    # step-1: presence detection (CLIP)
    ap.add_argument("--present_thr", type=float, default=0.25)  # CLIP similarity threshold: higher means stricter
    ap.add_argument("--min_cover", type=float, default=0.5)     # At least what proportion of images should be "present"
    # step-2: LangSAM mask
    ap.add_argument("--box_thr", type=float, default=0.25)
    ap.add_argument("--text_thr", type=float, default=0.25)

    # Flux
    ap.add_argument("--flux_model", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    ap.add_argument("--guidance", type=float, default=45.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")

    # save options
    ap.add_argument("--save_masks", action="store_true")
    args = ap.parse_args()

    # adaptive import
    if args.domain == "face":
        from utils_faces import SUBTLE_FEATURES, FeatureSpec, FACE_OPTIONAL_FIDS as OPTIONAL_FIDS
    else:
        from utils_arts import SUBTLE_FEATURES, FeatureSpec, ART_STYLE_OPTIONAL_FIDS as OPTIONAL_FIDS

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_masks:
        os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "masks_all"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = make_rng(args.key, args.nonce)

    image_paths = list_images(args.input_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {args.input_dir}")

    # ---------- Step 1: Determine which features are present (CLIP) ----------
    if args.domain == "face":
        clip_model, clip_processor = build_clip(device)

        present_fids, cover, mean_sim = detect_present_fids(
            image_paths=image_paths,
            clip_model=clip_model,
            clip_processor=clip_processor,
            SUBTLE_FEATURES=SUBTLE_FEATURES,
            OPTIONAL_FIDS=OPTIONAL_FIDS,
            device=device,
            present_thr=args.present_thr,
            min_cover=args.min_cover,
        )
    else:
        # The style primitives do not perform "existence" checks: all are allowed as candidates.
        present_fids = [f.fid for f in SUBTLE_FEATURES]
        cover = {fid: 1.0 for fid in present_fids}
        mean_sim = {fid: 1.0 for fid in present_fids}

    feature_map = {f.fid: f for f in SUBTLE_FEATURES}

    # ---------- Step 2: Randomly select a feature and generate a mask (LangSAM). ----------
    cand_fids = present_fids[:]  # copy
    rng.shuffle(cand_fids)

    sam = LangSAM()

    chosen_fid = None
    chosen_feature = None
    text_prompt = None
    sample_mask = None

    # To generate a "more relevant" prompt, a sample image is used first to obtain a sample mask.
    sample_idx = rng.randrange(len(image_paths))
    sample_path = image_paths[sample_idx]
    sample_img = pil_load_rgb(sample_path)

    for fid_try in cand_fids:
        feat = feature_map[fid_try]
        tp = rng.choice(feat.langsam_prompts)

        k = getattr(feat, "topk", 1)
        if tp == "__PATCH__":
            sm = patch_select_mask(sample_img, rng)
        else:
            sm = langsam_select_mask(
                sam=sam,
                image_path=sample_path,
                text_prompt=tp,
                box_threshold=args.box_thr,
                text_threshold=args.text_thr,
                topk=k,
            )

        # Key point: If topk=1 and all masks are > 0.4, it will return None -> move to the next feature.
        if sm is None or sm.sum() == 0:
            continue

        chosen_fid = fid_try
        chosen_feature = feat
        text_prompt = tp
        sample_mask = sm
        break

    if chosen_fid is None:
        raise RuntimeError("No feature yields a valid mask on the sample image. "
                        "Try lowering box/text thresholds or expanding feature pool.")

    edit_prompt = build_edit_prompt(
        feature=chosen_feature,
        rng=rng,
        sample_image=sample_img,
        sample_mask=sample_mask,
    )

    # build Flux pipe
    pipe = build_flux_pipe(args.flux_model, device=device, dtype=args.dtype)

    # meta
    meta = {
        "key_used": bool(args.key), "nonce": args.nonce, "chosen_feature": chosen_fid, "edit_prompt": edit_prompt,
        "presence": {"present_thr": args.present_thr, "min_cover": args.min_cover, 
            "present_fids": present_fids, "cover": cover, "mean_sim": mean_sim,},
        "langsam": {"box_thr": args.box_thr, "text_thr": args.text_thr,},
        "flux": {"model": args.flux_model, "guidance": args.guidance, "steps": args.steps,
            "max_seq_len": args.max_seq_len, "dtype": args.dtype,},
        "skipped": [],
    }

    # ---------- Apply to all images ----------
    for idx, p in enumerate(tqdm(image_paths, desc="Editing")):
        img = pil_load_rgb(p)
        # Each image has a unique seed (even with the same nonce, the results will vary).
        seed = int(hashlib.sha256(f"{args.key}::{args.nonce}::{os.path.basename(p)}::{idx}".encode()).hexdigest(), 16) % (2**32)

        if text_prompt == "__PATCH__":
            per_rng = random.Random(seed)  # The patch position is different in each image.
            mask = patch_select_mask(img, per_rng)
        else:
            save_prefix = os.path.join(args.output_dir, "masks_all", str(idx)) if args.save_masks else None
            mask = langsam_select_mask(
                sam=sam,
                image_path=p,
                text_prompt=text_prompt,
                box_threshold=args.box_thr,
                text_threshold=args.text_thr,
                topk=k,
                save_prefix = save_prefix,
            )
        if mask is None or mask.sum() == 0:
            meta["skipped"].append(os.path.basename(p))
            continue

        if args.save_masks:
            mpath = os.path.join(args.output_dir, "masks", os.path.basename(p))
            cv2.imwrite(mpath, (mask.astype(np.uint8) * 255))

        flux_mask = to_uint8_mask(mask)

        out = flux_edit_one(
            pipe=pipe,
            image=img,
            mask=flux_mask,
            prompt=edit_prompt,
            seed=seed,
            guidance_scale=args.guidance,
            steps=args.steps,
            max_sequence_length=args.max_seq_len,
        )

        out_name = os.path.splitext(os.path.basename(p))[0] + ".png"
        out.save(os.path.join(args.output_dir, out_name))

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Chosen feature: {chosen_fid}")
    print(f"Chosen text prompt: {text_prompt}")
    print(f"Edit prompt: {edit_prompt}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
