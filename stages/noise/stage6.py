"""Optimization 6 on stage 5: captured layer inputs keep their dtype. dust_noise.py stored a float32 copy of every
bfloat16 layer input (the inputs of both c_proj); here they are stored as they are and widened when read.
"""
from dust_noise import Capture
from stages.noise import stage5

Settings = stage5.Settings


class Inputs(dict):
    """Layer inputs as captured, widened to float32 when read."""

    def __getitem__(self, name):
        value = super().__getitem__(name)
        return value.float() if value.is_floating_point() else value


class LeanCapture(Capture):
    def __init__(self, *args):
        super().__init__(*args)
        self.inputs = Inputs()

    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0]
                self.outputs[name] = output

        return hook


class Boost(stage5.Boost):
    capture_class = LeanCapture

    def __init__(self, model, settings, seed, rank, world, split=1, group=None):
        super().__init__(model, settings, seed, rank, world, split, group)
        self.capture.close()  # the same capture and generators, of capture_class
        self.capture = self.capture_class(model, seed + 1000 * (rank // split), self.part, split)
        self.shared_generator = self.capture.shared_generator


class Cut(Boost, stage5.Cut):
    pass
