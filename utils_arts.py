#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Art-style feature bank for WikiArt / artistic paintings.

Goal: protect "style" rather than "objects".
These features are **local style primitives** (stroke / texture / linework / tonal treatment)
that can be applied in a small masked region (patch-based or background-based).

Recommended simplest integration:
- Use patch-based masks (no LangSAM) by setting langsam_prompts=["__PATCH__"] for all specs.
- In run_edit.py: if text_prompt == "__PATCH__", create a random rectangular mask (patch) and edit it.
"""

from typing import List
from dataclasses import dataclass

# Reuse the same FeatureSpec schema as your face utils.py.
# If you prefer, you can also `from utils import FeatureSpec` and remove this class.
@dataclass
class FeatureSpec:
    fid: str
    clip_phrases: List[str]
    caption_keywords: List[str]
    langsam_prompts: List[str]          # use "__PATCH__" to indicate patch-mask policy
    edit_templates: List[str]
    topk: int = 1
    too_obvious: bool = False


# For art-style we do NOT rely on presence detection.
# Keep OPTIONAL_FIDS empty so your pipeline can treat all as "always candidates".
ART_STYLE_OPTIONAL_FIDS = set()

# Keep text simple and consistent with FluxFill "instruction editing".
ART_KEEP_ALL = (
    "keep composition, subject, perspective, and everything outside the mask unchanged."
)

ART_SUBJECT = "High-resolution painting of the same artwork"

# NOTE:
# - We intentionally avoid object-specific edits (e.g., "add a vase").
# - All specs use "__PATCH__" so the mask is a patch and will always exist across a painter's works.
# - Templates are "change/replace" and direct, without "tiny/small/subtle/slightly" wording.

SUBTLE_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        fid="stroke_direction",
        clip_phrases=["a painting with visible brushstrokes", "an oil painting", "a canvas painting"],
        caption_keywords=["painting", "brushstrokes", "canvas"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change brushstroke direction to diagonal, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change brushstroke direction to horizontal, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change brushstroke direction to vertical, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change brushstroke direction to swirling, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="stroke_thickness",
        clip_phrases=["a painting with visible brushstrokes", "an oil painting"],
        caption_keywords=["painting", "brushstrokes"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change brushstroke thickness to thick, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change brushstroke thickness to thin, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="stroke_length",
        clip_phrases=["a painting with visible brushstrokes", "a canvas painting"],
        caption_keywords=["painting", "brushstrokes"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change brushstroke length to short, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change brushstroke length to long, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="impasto_texture",
        clip_phrases=["an oil painting", "a thick paint texture", "impasto painting"],
        caption_keywords=["painting", "impasto", "texture"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change paint texture to impasto, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove impasto texture and make paint texture smooth, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="canvas_grain",
        clip_phrases=["a canvas painting", "canvas texture", "painting surface"],
        caption_keywords=["canvas", "texture", "painting"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change canvas texture to rough, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change canvas texture to smooth, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="paper_grain",
        clip_phrases=["a watercolor painting", "paper texture", "illustration on paper"],
        caption_keywords=["paper", "watercolor", "texture"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change paper texture to rough, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change paper texture to smooth, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="edge_style",
        clip_phrases=["a painting", "illustration", "artwork"],
        caption_keywords=["edges", "painting"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change edge style to hard edges, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change edge style to soft edges, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="outline_linework",
        clip_phrases=["a painting", "illustration", "linework artwork"],
        caption_keywords=["outline", "linework", "drawing"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, add ink-like outlines around shapes, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove outlines and make edges painterly, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="crosshatching_shading",
        clip_phrases=["a drawing", "a sketch", "crosshatching shading"],
        caption_keywords=["crosshatching", "shading", "drawing"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change shading style to crosshatching, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove crosshatching and make shading smooth, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="hatching_angle",
        clip_phrases=["a drawing", "a sketch", "hatching shading"],
        caption_keywords=["hatching", "shading"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change hatching angle to 30-degree, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change hatching angle to 45-degree, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change hatching angle to 60-degree, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="stippling_texture",
        clip_phrases=["a stippling drawing", "pointillism", "dotted texture painting"],
        caption_keywords=["stippling", "pointillism", "dots"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change texture to stippling, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove stippling and make texture smooth, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="watercolor_bleed",
        clip_phrases=["a watercolor painting", "watercolor wash", "ink wash painting"],
        caption_keywords=["watercolor", "wash", "bleed"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change paint edges to watercolor bleed, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove watercolor bleed and make edges crisp, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="flat_color_blocks",
        clip_phrases=["a poster illustration", "color blocking artwork", "flat color illustration"],
        caption_keywords=["poster", "flat color", "blocks"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change shading to flat color blocks, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove flat color blocks and make shading smooth, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="color_temperature",
        clip_phrases=["a painting", "an artwork", "illustration"],
        caption_keywords=["color", "tone"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change color temperature to warmer, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change color temperature to cooler, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="saturation_level",
        clip_phrases=["a painting", "an artwork", "illustration"],
        caption_keywords=["color", "saturation"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change saturation to higher, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change saturation to lower, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="contrast_level",
        clip_phrases=["a painting", "an artwork", "illustration"],
        caption_keywords=["contrast", "tone"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change contrast to higher, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change contrast to lower, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="grain_texture",
        clip_phrases=["a painting", "grainy artwork", "textured artwork"],
        caption_keywords=["grain", "texture"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, add grain texture, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove grain texture and make surface clean, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="glaze_effect",
        clip_phrases=["a painting", "glazing in painting", "transparent glaze"],
        caption_keywords=["glaze", "painting"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, add glazing effect, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove glazing effect, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="chromatic_fringe",
        clip_phrases=["a painting", "chromatic aberration effect", "color fringing"],
        caption_keywords=["chromatic", "fringe"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, add chromatic edge fringing, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove chromatic edge fringing, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="highlight_style",
        clip_phrases=["a painting", "artwork highlights", "specular highlights"],
        caption_keywords=["highlight", "painting"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change highlights to sharp, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change highlights to soft, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="shadow_style",
        clip_phrases=["a painting", "artwork shadows", "cell shading"],
        caption_keywords=["shadow", "shading"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change shadows to cell shading, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove cell shading and make shadows smooth, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="line_density",
        clip_phrases=["a drawing", "sketch", "dense linework"],
        caption_keywords=["linework", "sketch"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change linework density to higher, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change linework density to lower, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="ink_wash",
        clip_phrases=["ink wash painting", "sumi-e painting", "ink painting"],
        caption_keywords=["ink", "wash"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change paint to ink wash style, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, remove ink wash and make paint opaque, {ART_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="color_blending",
        clip_phrases=["a painting", "smooth color blending", "blended brushwork"],
        caption_keywords=["blending", "painting"],
        langsam_prompts=["__PATCH__"],
        edit_templates=[
            f"{ART_SUBJECT}, change color blending to smooth, {ART_KEEP_ALL}",
            f"{ART_SUBJECT}, change color blending to broken color, {ART_KEEP_ALL}",
        ],
    ),
]
