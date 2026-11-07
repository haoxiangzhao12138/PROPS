"""Qwen-Image-Edit-2511 as run for the reference row: frame 0 and the last two keyframes as reference images (its maximum
of three), and the whole story as text.

    props generate --adapter props.adapters.qwen_edit -- --path Qwen/Qwen-Image-Edit-2511
Needs diffusers with QwenImageEditPlusPipeline and one GPU with about 60 GB.
"""
import argparse


def refs(ctx, n):
    """Indices of the history frames shown as images: all of them if they fit, else frame 0 and the latest ones."""
    k = len(ctx["history"])
    if n <= 0 or n >= k:
        return list(range(k))
    return [k - 1] if n == 1 else [0] + list(range(k - n + 1, k))


def story(ctx, ks, name):
    pos = {k: i for i, k in enumerate(ks)}
    out = []
    for k in range(len(ctx["history"])):
        line = "Frame 0 (start)" if k == 0 else f"Keyframe {k} ({ctx['dt_text'][k - 1]}): {ctx['scripts'][k - 1]}"
        out.append(line + (f" [{name(pos[k])}]" if k in pos else " [not shown]"))
    return out


def edit_instruction(ctx, ks, name=lambda i: f"Picture {i + 1}"):
    """The instruction multi-reference editors receive: the story with the shown frames named, then the new step."""
    return ("The input pictures are keyframes of a storyboard made from real footage. Story so far:\n"
            + "\n".join(story(ctx, ks, name))
            + f"\nNow draw keyframe {ctx['j']} ({ctx['dt_text'][-1]}): {ctx['script']}\n"
            f"The script only states what changed. Keep everything it does not mention (people, objects, place, "
            f"lighting, camera style) as in {name(len(ks) - 1)}, and keep people, objects and places looking the "
            "same as in the earlier keyframes. Output one photorealistic frame.")


def fit(size, area=1024 * 1024, mult=16):
    w, h = size
    s = (area / (w * h)) ** 0.5
    return max(mult, round(w * s / mult) * mult), max(mult, round(h * s / mult) * mult)


class Model:
    def __init__(self, args):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        ap = argparse.ArgumentParser(prog="qwen_edit")
        ap.add_argument("--path", default="Qwen/Qwen-Image-Edit-2511")
        ap.add_argument("--refs", type=int, default=3)
        ap.add_argument("--steps", type=int, default=40)
        ap.add_argument("--cfg", type=float, default=4.0)
        self.a = ap.parse_args(args)
        self.torch = torch
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(self.a.path, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, ctx):
        ks = refs(ctx, self.a.refs)
        w, h = fit(ctx["size"])
        with self.torch.inference_mode():
            return self.pipe(image=[ctx["history"][k] for k in ks], prompt=edit_instruction(ctx, ks), negative_prompt=" ",
                             true_cfg_scale=self.a.cfg, guidance_scale=1.0, num_inference_steps=self.a.steps,
                             width=w, height=h, generator=self.torch.Generator("cuda").manual_seed(ctx["seed"])).images[0]
