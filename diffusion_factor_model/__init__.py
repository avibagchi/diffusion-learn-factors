"""
Diffusion Factor Model: A diffusion-based framework for financial factor modeling
"""

__version__ = "0.1.0"

_LEGACY = {"Unet", "GaussianDiffusion", "Trainer", "GaussianLatentSampler2D_Finance"}
_DESCRIPTOR = {
    "DescriptorFactorDiffusion",
    "DescriptorReturnDataset",
    "DescriptorScoreNetwork",
    "DescriptorTrainer",
    "generate_descriptor_dataset",
    "load_descriptor_dataset",
    "save_descriptor_dataset",
}


def __getattr__(name):
    if name in _DESCRIPTOR:
        from diffusion_factor_model import descriptor_score_model as mod

        return getattr(mod, name)
    if name in _LEGACY:
        from diffusion_factor_model import diffusion_factor_model as mod

        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") 