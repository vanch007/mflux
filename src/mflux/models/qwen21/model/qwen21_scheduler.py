import mlx.core as mx
import numpy as np

from mflux.models.common.schedulers import register_contrib
from mflux.models.common.schedulers.base_scheduler import BaseScheduler
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler


class ViggleTurboScheduler(BaseScheduler):
    SIGMA_NODES = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
    SCHEDULES = {
        6: SIGMA_NODES,
        8: (1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125),
    }

    def __init__(self, config):
        self.config = config
        steps = config.num_inference_steps
        if steps not in self.SCHEDULES:
            raise ValueError(
                f"viggle_turbo supports the published 6-step or 8-step schedules, got {steps}. "
                "Use --steps 6 (default) or --steps 8 for dense text."
            )
        self._sigmas = LinearScheduler.shift_sigmas(
            mx.array(self.SCHEDULES[steps], dtype=mx.float32), config, use_terminal=False
        )
        self._timesteps = mx.arange(steps, dtype=mx.float32)

    @staticmethod
    def check_args(parser, args) -> None:
        if args.scheduler != "viggle_turbo":
            return
        if args.steps not in ViggleTurboScheduler.SCHEDULES:
            parser.error("--scheduler viggle_turbo requires --steps 6 or --steps 8.")
        if not args.lora_paths and not getattr(args, "lora_preset", None):
            print("⚠️  --scheduler viggle_turbo needs its distilled adapter for Turbo results.")

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return (latents.astype(mx.float32) + noise.astype(mx.float32) * dt).astype(latents.dtype)


register_contrib(ViggleTurboScheduler, "viggle_turbo")
register_contrib(ViggleTurboScheduler, "ViggleTurboScheduler")


class Qwen21TurboScheduler(BaseScheduler):
    """Fixed 8-step schedule of the Qwen-Image-2.1-Turbo checkpoint.

    The checkpoint saves its sigma nodes as `sample_sigmas` in model_index.json, and its
    scheduler config turns dynamic shifting off (shift 1.0, no shift_terminal). So this
    scheduler uses the nodes raw, with no resolution shift and no terminal rescale. Other step
    counts sample the same curve at evenly spaced points. step() is the same first-order Euler update LinearScheduler uses.

    The Turbo model selects this schedule for the default "linear" scheduler name. See
    for_model().
    """

    SIGMA_NODES = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)

    def __init__(self, config):
        self.config = config
        steps = config.num_inference_steps
        nodes = [*self.SIGMA_NODES, 0.0]
        if steps == len(self.SIGMA_NODES):
            sigmas = nodes
        else:
            sigmas = np.interp(np.linspace(0.0, 1.0, steps + 1), np.linspace(0.0, 1.0, len(nodes)), nodes)
        self._sigmas = mx.array(sigmas, dtype=mx.float32)
        self._timesteps = mx.arange(steps, dtype=mx.float32)

    @staticmethod
    def is_turbo(model_config) -> bool:
        from mflux.models.common.config.model_config import ModelConfig

        turbo = ModelConfig.qwen_image_21_turbo().model_name
        return turbo in (model_config.model_name, model_config.base_model)

    @staticmethod
    def check_generate(model_config, num_inference_steps: int, guidance: float, scheduler: str = "linear") -> None:
        if model_config is not None and not Qwen21TurboScheduler.is_turbo(model_config):
            return
        if guidance != 1:
            raise ValueError(f"Qwen-Image-2.1-Turbo runs without CFG. Got guidance={guidance}. Use guidance=1.")
        if scheduler not in ("linear", "qwen_turbo", "qwen21_turbo", "lightning", "qwen_lightning"):
            raise ValueError(f"Qwen-Image-2.1-Turbo runs its own schedule. Got scheduler={scheduler!r}. Use 'qwen_turbo' or 'linear'.")

    @staticmethod
    def default_steps(model_config) -> int:
        from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS

        if Qwen21TurboScheduler.is_turbo(model_config):
            return len(Qwen21TurboScheduler.SIGMA_NODES)
        return MODEL_INFERENCE_STEPS["qwen-image-2.1"]

    @staticmethod
    def for_model(model_config, scheduler: str) -> str:
        if scheduler == "linear" and Qwen21TurboScheduler.is_turbo(model_config):
            return "qwen21_turbo"
        return scheduler

    @staticmethod
    def check_args(parser, args, model_config=None) -> None:
        if model_config is not None and not Qwen21TurboScheduler.is_turbo(model_config):
            return
        nodes = len(Qwen21TurboScheduler.SIGMA_NODES)
        if args.steps != nodes and not parser._option_was_provided("--steps"):
            args.steps = nodes
        if args.guidance is not None and args.guidance != 1:
            parser.error(f"Qwen-Image-2.1-Turbo runs without CFG. Got --guidance {args.guidance}. Use --guidance 1.")
        if args.scheduler == "viggle_turbo":
            parser.error("viggle_turbo is the schedule of a LoRA for the base Qwen-Image-2.1. Do not use it with the Turbo model.")

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return (latents.astype(mx.float32) + noise.astype(mx.float32) * dt).astype(latents.dtype)


register_contrib(Qwen21TurboScheduler, "qwen21_turbo")
register_contrib(Qwen21TurboScheduler, "Qwen21TurboScheduler")
register_contrib(Qwen21TurboScheduler, "qwen_turbo")
register_contrib(Qwen21TurboScheduler, "QwenTurboScheduler")

QwenTurboScheduler = Qwen21TurboScheduler


class QwenLightningScheduler(BaseScheduler):
    """Official 4-step sampling schedule for Qwen-Image-2.1-Lightning."""
    SIGMA_NODES = (1.0, 0.75, 0.5, 0.25)

    def __init__(self, config):
        self.config = config
        steps = config.num_inference_steps
        if steps != 4:
            raise ValueError(
                f"lightning supports the official 4-step schedule, got {steps}. "
                "Use --steps 4 for Qwen-Image-2.1-Lightning."
            )
        self._sigmas = LinearScheduler.shift_sigmas(
            mx.array(self.SIGMA_NODES, dtype=mx.float32), config, use_terminal=False
        )
        self._timesteps = mx.arange(steps, dtype=mx.float32)

    @staticmethod
    def check_args(parser, args) -> None:
        if args.scheduler not in ("lightning", "qwen_lightning"):
            return
        if args.steps != 4:
            parser.error("--scheduler lightning requires --steps 4.")

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = self._sigmas[timestep + 1] - self._sigmas[timestep]
        return (latents.astype(mx.float32) + noise.astype(mx.float32) * dt).astype(latents.dtype)


register_contrib(QwenLightningScheduler, "lightning")
register_contrib(QwenLightningScheduler, "qwen_lightning")
register_contrib(QwenLightningScheduler, "QwenLightningScheduler")

