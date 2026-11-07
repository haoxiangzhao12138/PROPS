"""Every prompt PROPS uses, verbatim from the reference runs.

Generation: `generation_prompt` is the instruction that interleaved and multi-image models receive.
Scoring: CHOICE (Follow), SEE (Exec), and the Reuse grounding prompts GROUND, OWN_GROUND, OWN_CROP, OWN_STRICT.
Changing any of them changes the scores, so a submission to the leaderboard must use them as they are.
"""

# ---------------------------------------------------------------- generation

def step_text(item, j, current=False):
    """One script line. On the step being drawn, `draw_script` (Reuse: the script that names the look) replaces it."""
    st = item["steps"][j - 1]
    text = st.get("draw_script", st["script"]) if current else st["script"]
    return f"Keyframe {j} ({st['time']}): {text}"


def generation_prompt(item, j):
    prev = "\n".join(step_text(item, k) for k in range(1, j))
    return ("You are drawing the next keyframe of a storyboard made from real footage. The images given are "
            "frame 0 and the earlier keyframes, in order. Each keyframe follows a script that only states what "
            "changed; anything the script does not change (people, objects, places, their look) must stay "
            "exactly as in the earlier frames.\n"
            + (f"Earlier scripts:\n{prev}\n" if prev else "")
            + f"Now draw ONLY {step_text(item, j, current=True)}\n"
            "Output one photorealistic frame with the same camera style and aspect ratio as frame 0.")


# ---------------------------------------------------------------- judge (Gemma-4-31B-it)

CHOICE_SYS = "You compare objects in images carefully."
CHOICE = """Images 1 and 2 are close-ups of two different versions of {kind}. Image 3 is a scene. Which version appears in
image 3? Look at colours, pattern, printed details and shape. Reply with JSON only: {{"answer": "1", "2" or "neither"}}"""

JSYS = "You are a careful visual judge. Reply with one JSON object only."
SEE = """Answer each question about this image with "yes", "no" or "unsure" (unsure only if it truly cannot be
seen). Questions:
{qs}
Return {{"answers": ["yes" | "no" | "unsure", ...]}} with one answer per question, in order."""


def numbered(qs):
    return "\n".join(f"{n + 1}. {x['q']}" for n, x in enumerate(qs))


# ---------------------------------------------------------------- Reuse grounding (Qwen3.8-27B, with the judge)

GROUND_SYS = "You locate objects in images precisely."
GROUND = """The image is a frame from a video. Target object: {name} ({look}; {where}).
Locate it and reply with JSON only:
{{"found": true if this exact object is clearly visible (one instance, not mostly hidden, not cut off by the border),
 "bbox": [x1, y1, x2, y2] tight box around it, coordinates normalised to 0-1000 of the image width / height,
 "polygon": [[x, y], ...] 10-24 points tracing its outline, same coordinates,
 "kind": what it is in 1-3 words{more}}}"""
OWN_GROUND = """The image is a frame from a video. Find the {name} in it (it should be {look}; {where}). It may be thin,
small, partly hidden or partly cut off, and may look different from that description. Reply with JSON only:
{{"found": false only if there is no {name} at all, "bbox": [x1, y1, x2, y2] tight box around the {name} (the most
likely one if several), coordinates normalised to 0-1000 of the image width / height}}"""

CHECK_SYS = "You look at images carefully."
OWN_CROP = """Is the main object in this image a {kind} (any colour or pattern)? Reply with JSON only: {{"answer": "yes" | "no"}}"""
OWN_STRICT = """Look at the main object in this image. Is it a {kind}? Answer "no" if it is a different kind of object
(for example a cloth, a rack, a mat, a bottle, a box or part of the background) even if it has a similar colour,
shape or pattern. Reply with JSON only: {{"answer": "yes" | "no"}}"""
