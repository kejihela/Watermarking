from typing import List, Dict
from dataclasses import dataclass, field

# -----------------------------
# Feature Bank (face-focused)
# -----------------------------
@dataclass
class FeatureSpec:
    fid: str
    # for CLIP "is this feature present"
    clip_phrases: List[str]
    caption_keywords: List[str]
    # LangSAM text prompt(s)
    langsam_prompts: List[str]
    # FluxFill prompt templates (Subject + Action + Style + Context)
    edit_templates: List[str]
    # how many instances to keep (e.g., eyes/eyebrows/cheeks often 2)
    topk: int = 1
    too_obvious: bool = False

    # Placeholder banks for edit_templates, e.g. {color}, {side}, ...
    # Put feature-specific values here so build_edit_prompt can be generic.
    slot_values: Dict[str, List[str]] = field(default_factory=dict)

    # Optional simple policies per slot (keep build_edit_prompt generic)
    # e.g. {"color": "different_from_mask"}
    slot_policies: Dict[str, str] = field(default_factory=dict)



# Only items that "may be present" undergo presence detection; all others are added to the candidate pool by default.
FACE_OPTIONAL_FIDS = {
    "eyeliner",
    "under_eye_area",
    "teeth",
    "freckles_cheeks",
    "beauty_mark_cheek",
    "beard_style",
    "mustache_style",
    "goatee_style",
    "sideburns",
    "stubble",
    "bangs",
}

# helper strings (keep them simple; avoid "tiny/small/subtle/slightly")
PORTRAIT_KEEP_ALL = "keep face identity, expression, hair, lighting unchanged, natural skin texture, keep everything else unchanged."
PORTRAIT_KEEP_NO_HAIR = "keep face identity, expression, lighting unchanged, natural skin texture, keep everything else unchanged."
PORTRAIT_KEEP_NO_LIPS = "keep face identity, expression, lighting unchanged, natural skin texture, keep everything else unchanged."

# slot policies
POLICY_DIFFERENT_FROM_MASK = "different_from_mask"

# Slot candidate banks (direct/simple words; no 'tiny/small/subtle/slightly')
SLOT_COLOR_EYE = ["blue", "emerald green", "hazel", "amber", "gray", "violet"]
SLOT_PUPIL_SIZE = ["larger", "smaller"]
SLOT_SHAPE = ["circle", "star", "crescent", "diamond", "heart"]

SLOT_LASH_STYLE = ["thicker", "longer", "curlier", "sparser"]
SLOT_LASH_LENGTH = ["longer", "shorter"]
SLOT_LINER_STYLE = ["winged", "cat-eye", "tightline", "smoky"]

SLOT_BROW_THICKNESS = ["thicker", "thinner"]
SLOT_BROW_COLOR = ["black", "dark brown", "light brown", "auburn"]

SLOT_NOSTRIL_SHAPE = ["narrower", "wider", "more oval", "more round"]

SLOT_LIP_COLOR = ["red", "rose", "coral", "plum", "nude", "brick red"]
SLOT_TOOTH_SHADE = ["A1", "B1", "A2", "B2"]

SLOT_BLUSH_COLOR = ["rose", "peach", "coral", "pink"]
SLOT_SIDE = ["left", "right"]

SLOT_HAIR_COLOR = ["black", "brown", "blonde", "platinum blonde", "red", "auburn"]
SLOT_HAIR_TEXTURE = ["curly", "straight", "wavy"]
SLOT_BANGS_STYLE = ["straight", "curtain bangs", "side-swept"]

SLOT_BEARD_STYLE = ["full beard", "short boxed beard", "circle beard", "ducktail"]
SLOT_BEARD_TEXTURE = ["curly", "straight", "wavy"]
SLOT_MUSTACHE_STYLE = ["handlebar", "pencil", "chevron", "walrus"]
SLOT_GOATEE_STYLE = ["classic goatee", "anchor", "circle goatee", "soul patch"]
SLOT_SIDEBURN_LENGTH = ["shorter", "longer"]
SLOT_STUBBLE_DENSITY = ["denser", "lighter"]

SUBTLE_FEATURES: List[FeatureSpec] = [
    # ---------- Eyes ----------
    FeatureSpec(
        fid="pupil_iris",
        clip_phrases=["human eyes", "close-up portrait of a person", "iris and pupil"],
        caption_keywords=["eye", "eyes", "iris", "pupil"],
        langsam_prompts=["eyes", "eye"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change iris color to {{color}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change pupil color to {{color}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change iris color to {{color}} for both eyes, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"color": SLOT_COLOR_EYE},
        slot_policies={"color": POLICY_DIFFERENT_FROM_MASK},
    ),
    FeatureSpec(
        fid="pupil_size",
        clip_phrases=["human eyes", "close-up portrait", "pupil"],
        caption_keywords=["eye", "pupil"],
        langsam_prompts=["eyes"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change pupil size to {{pupil_size}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change pupil size to {{pupil_size}} for both eyes, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"pupil_size": SLOT_PUPIL_SIZE},
    ),
    FeatureSpec(
        fid="eye_catchlight",
        clip_phrases=["human eyes", "eye reflection", "catchlight in eyes"],
        caption_keywords=["eye", "reflection"],
        langsam_prompts=["eyes"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, add a catchlight in the eyes, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, replace the eye catchlight shape with {{shape}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"shape": SLOT_SHAPE},
    ),
    FeatureSpec(
        fid="eyelashes",
        clip_phrases=["eyelashes", "close-up portrait", "eye lashes"],
        caption_keywords=["eyelash", "eyelashes", "eye"],
        langsam_prompts=["eyelashes", "eyes"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change eyelashes to {{lash_style}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change eyelash length to {{lash_length}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"lash_style": SLOT_LASH_STYLE, "lash_length": SLOT_LASH_LENGTH},
    ),
    FeatureSpec(
        fid="eyeliner",
        clip_phrases=["eyeliner", "makeup eyeliner", "eye makeup"],
        caption_keywords=["eyeliner", "makeup", "eye"],
        langsam_prompts=["eyes", "eyelids"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, add eyeliner, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change eyeliner style to {{liner_style}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"liner_style": SLOT_LINER_STYLE},
    ),

    # ---------- Eyebrows ----------
    FeatureSpec(
        fid="eyebrows_shape",
        clip_phrases=["eyebrows", "portrait with eyebrows"],
        caption_keywords=["eyebrow", "eyebrows"],
        langsam_prompts=["eyebrows", "eyebrow"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change eyebrow shape to arched, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change eyebrow thickness to {{brow_thickness}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"brow_thickness": SLOT_BROW_THICKNESS},
    ),
    FeatureSpec(
        fid="eyebrows_color",
        clip_phrases=["eyebrows", "portrait with eyebrows"],
        caption_keywords=["eyebrow", "eyebrows"],
        langsam_prompts=["eyebrows"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change eyebrow color to {{brow_color}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change eyebrow tone to {{brow_color}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"brow_color": SLOT_BROW_COLOR},
    ),

    # ---------- Under-eye / skin near eyes ----------
    FeatureSpec(
        fid="under_eye_area",
        clip_phrases=["bags under eyes", "under-eye shadows", "dark circles under eyes"],
        caption_keywords=["under-eye", "eye", "face"],
        langsam_prompts=["under-eye area", "bags under eyes", "under eye"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, remove under-eye dark circles, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, add under-eye crease lines, {PORTRAIT_KEEP_ALL}",
        ],
    ),

    # ---------- Nose ----------
    FeatureSpec(
        fid="nose_tip",
        clip_phrases=["nose", "portrait face nose"],
        caption_keywords=["nose"],
        langsam_prompts=["nose tip", "nose"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change nose tip highlight to stronger, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change nose tip highlight to weaker, {PORTRAIT_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="nose_bridge",
        clip_phrases=["nose bridge", "nose", "portrait face"],
        caption_keywords=["nose"],
        langsam_prompts=["nose bridge", "nose"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change nose bridge highlight to stronger, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change nose bridge highlight to weaker, {PORTRAIT_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="nostrils",
        clip_phrases=["nostrils", "nose", "portrait face"],
        caption_keywords=["nose", "nostril"],
        langsam_prompts=["nostrils", "nose"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change nostril shape to {{nostril_shape}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"nostril_shape": SLOT_NOSTRIL_SHAPE},
    ),

    # ---------- Lips / mouth ----------
    FeatureSpec(
        fid="lips_color",
        clip_phrases=["lips", "mouth", "portrait lips"],
        caption_keywords=["lips", "mouth"],
        langsam_prompts=["lips", "mouth"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change lip color to {{lip_color}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change lipstick color to {{lip_color}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"lip_color": SLOT_LIP_COLOR},
    ),
    FeatureSpec(
        fid="lips_finish",
        clip_phrases=["lips", "mouth", "portrait lips"],
        caption_keywords=["lips", "mouth"],
        langsam_prompts=["lips"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change lips to glossier, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change lips to matte, {PORTRAIT_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="teeth",
        clip_phrases=["teeth visible", "smiling with teeth", "open mouth teeth"],
        caption_keywords=["teeth", "mouth"],
        langsam_prompts=["teeth", "open mouth"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, whiten the teeth, {PORTRAIT_KEEP_NO_LIPS}",
            f"Close-up photorealistic portrait of the same person, change tooth shade to {{tooth_shade}}, {PORTRAIT_KEEP_NO_LIPS}",
        ],
        slot_values={"tooth_shade": SLOT_TOOTH_SHADE},
    ),

    # ---------- Cheeks / skin ----------
    FeatureSpec(
        fid="cheeks_blush",
        clip_phrases=["cheeks", "face", "portrait face"],
        caption_keywords=["cheek", "face"],
        langsam_prompts=["cheeks", "cheek"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change cheek blush to {{blush_color}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, remove cheek blush, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"blush_color": SLOT_BLUSH_COLOR},
    ),
    FeatureSpec(
        fid="freckles_cheeks",
        clip_phrases=["freckles", "freckles on cheeks", "freckled face"],
        caption_keywords=["freckle", "freckles", "cheek"],
        langsam_prompts=["cheeks", "cheek"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, add freckles on the cheeks, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, remove freckles on the cheeks, {PORTRAIT_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="beauty_mark_cheek",
        clip_phrases=["beauty mark on cheek", "mole on cheek", "face mole"],
        caption_keywords=["mole", "beauty mark", "cheek"],
        langsam_prompts=["cheek", "face"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, add a beauty mark on the {{side}} cheek, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, remove the beauty mark on the {{side}} cheek, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"side": SLOT_SIDE},
    ),
    FeatureSpec(
        fid="forehead_wrinkles",
        clip_phrases=["forehead wrinkles", "forehead lines", "wrinkled forehead"],
        caption_keywords=["forehead", "wrinkle"],
        langsam_prompts=["forehead"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, add forehead wrinkles, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, remove forehead wrinkles, {PORTRAIT_KEEP_ALL}",
        ],
    ),
    FeatureSpec(
        fid="skin_pores",
        clip_phrases=["skin pores", "face skin texture", "high detail skin"],
        caption_keywords=["skin", "pores", "cheek"],
        langsam_prompts=["cheek", "nose"],
        topk=2,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, increase skin pores detail, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, smooth skin texture, {PORTRAIT_KEEP_ALL}",
        ],
    ),

    # ---------- Hair ----------
    FeatureSpec(
        fid="hair_color",
        clip_phrases=["hair", "hairstyle", "portrait hair"],
        caption_keywords=["hair"],
        langsam_prompts=["hair", "hairstyle"],
        topk=1,
        edit_templates=[
            f"Photorealistic portrait of the same person, change hair color to {{hair_color}}, {PORTRAIT_KEEP_NO_HAIR}",
            f"Photorealistic portrait of the same person, change hair color to {{hair_color}}, keep face identity, expression, lighting unchanged, keep everything else unchanged.",
        ],
        slot_values={"hair_color": SLOT_HAIR_COLOR},
    ),
    FeatureSpec(
        fid="hair_texture",
        clip_phrases=["hair", "hairstyle", "portrait hair"],
        caption_keywords=["hair"],
        langsam_prompts=["hair"],
        topk=1,
        edit_templates=[
            f"Photorealistic portrait of the same person, change hair texture to {{hair_texture}}, {PORTRAIT_KEEP_NO_HAIR}",
            f"Photorealistic portrait of the same person, change hairstyle to {{hair_texture}}, {PORTRAIT_KEEP_NO_HAIR}",
        ],
        slot_values={"hair_texture": SLOT_HAIR_TEXTURE},
    ),
    FeatureSpec(
        fid="bangs",
        clip_phrases=["bangs", "fringe hair", "hair bangs"],
        caption_keywords=["bangs", "hair", "fringe"],
        langsam_prompts=["bangs", "fringe hair"],
        topk=1,
        edit_templates=[
            f"Photorealistic portrait of the same person, add bangs, {PORTRAIT_KEEP_NO_HAIR}",
            f"Photorealistic portrait of the same person, change bangs to {{bangs_style}}, {PORTRAIT_KEEP_NO_HAIR}",
        ],
        slot_values={"bangs_style": SLOT_BANGS_STYLE},
    ),
    FeatureSpec(
        fid="hairline",
        clip_phrases=["hairline", "forehead hairline", "portrait hairline"],
        caption_keywords=["hairline", "forehead", "hair"],
        langsam_prompts=["hairline", "forehead hairline"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change hairline to higher, {PORTRAIT_KEEP_NO_HAIR}",
            f"Close-up photorealistic portrait of the same person, change hairline to lower, {PORTRAIT_KEEP_NO_HAIR}",
        ],
    ),

    # ---------- Facial hair ----------
    FeatureSpec(
        fid="beard_style",
        clip_phrases=["beard", "facial hair", "full beard"],
        caption_keywords=["beard", "facial hair"],
        langsam_prompts=["beard", "facial hair"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change beard style to {{beard_style}}, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change beard texture to {{beard_texture}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"beard_style": SLOT_BEARD_STYLE, "beard_texture": SLOT_BEARD_TEXTURE},
    ),
    FeatureSpec(
        fid="mustache_style",
        clip_phrases=["mustache", "moustache", "facial hair mustache"],
        caption_keywords=["mustache", "moustache"],
        langsam_prompts=["mustache", "moustache"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change mustache style to {{mustache_style}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"mustache_style": SLOT_MUSTACHE_STYLE},
    ),
    FeatureSpec(
        fid="goatee_style",
        clip_phrases=["goatee", "chin goatee", "facial hair goatee"],
        caption_keywords=["goatee", "chin"],
        langsam_prompts=["goatee", "chin beard"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change the goatee to curly, {PORTRAIT_KEEP_ALL}",
            f"Close-up photorealistic portrait of the same person, change goatee style to {{goatee_style}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"goatee_style": SLOT_GOATEE_STYLE},
    ),
    FeatureSpec(
        fid="sideburns",
        clip_phrases=["sideburns", "facial hair sideburns"],
        caption_keywords=["sideburns"],
        langsam_prompts=["sideburns"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change sideburn length to {{sideburn_length}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"sideburn_length": SLOT_SIDEBURN_LENGTH},
    ),
    FeatureSpec(
        fid="stubble",
        clip_phrases=["stubble", "five o'clock shadow", "facial stubble"],
        caption_keywords=["stubble", "beard"],
        langsam_prompts=["stubble", "five o'clock shadow"],
        topk=1,
        edit_templates=[
            f"Close-up photorealistic portrait of the same person, change stubble density to {{stubble_density}}, {PORTRAIT_KEEP_ALL}",
        ],
        slot_values={"stubble_density": SLOT_STUBBLE_DENSITY},
    ),
]