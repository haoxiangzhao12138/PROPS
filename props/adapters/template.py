"""Template for a new system. Copy this file, fill in __init__ and generate, then run

    props generate --adapter my_model.py -- --path /models/my-model

generate(ctx) receives one step (see props/generate.py for every field) and returns a PIL image. Give the model
the full history in its native input format, up to the number of images it accepts. An interleaved model should
see frame 0, keyframe 1 with script 1, and so on, ending with the script of the step to draw:

    parts = [ctx["history"][0]]
    for k in range(1, len(ctx["history"])):
        parts += [f"Keyframe {k} ({ctx['dt_text'][k - 1]}): {ctx['scripts'][k - 1]}", ctx["history"][k]]
    parts += [f"Now draw keyframe {ctx['j']} ({ctx['dt_text'][-1]}): {ctx['script']}"]

A model that takes one prompt and a list of images can use ctx["prompt"] and ctx["history"] instead. Seed the
sampler with ctx["seed"] so that a rerun gives the same frame. The output is resized to ctx["size"] when saved.
"""
import argparse


class Model:
    def __init__(self, args):
        ap = argparse.ArgumentParser(prog="my_model")
        ap.add_argument("--path", required=True)
        self.a = ap.parse_args(args)
        self.model = None  # load the model here

    def generate(self, ctx):
        raise NotImplementedError("draw keyframe ctx['j'] from ctx['history'] and the scripts")
