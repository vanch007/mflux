import mlx.core as mx

from mflux.models.common.schedulers import register_contrib
from mflux.models.common.schedulers.base_scheduler import BaseScheduler
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler


class QwenTurboScheduler(BaseScheduler):
    """Official 8-step sampling schedule for Qwen-Image-2.1-Turbo.

    Uses the exact trained sample_sigmas from the model_index.json without
    dynamic shifting or terminal scaling, matching official Diffusers PR #14950.
    """
    OFFICIAL_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)

    def __init__(self, config):
        self.config = config
        steps = config.num_inference_steps
        if steps != 8:
            raise ValueError(
                f"qwen_turbo supports the official 8-step schedule, got {steps}. "
                "Use --steps 8 for official Qwen-Image-2.1-Turbo."
            )
        sigmas_list = list(self.OFFICIAL_SIGMAS) + [0.0]
        self._sigmas = mx.array(sigmas_list, dtype=mx.float32)
        self._timesteps = self._sigmas[:-1] * 1000.0

    @staticmethod
    def check_args(parser, args) -> None:
        if args.scheduler != "qwen_turbo":
            return
        if args.steps != 8:
            parser.error("--scheduler qwen_turbo requires --steps 8.")

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array:
        dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
        return latents + noise.astype(latents.dtype) * dt


register_contrib(QwenTurboScheduler, "qwen_turbo")
register_contrib(QwenTurboScheduler, "QwenTurboScheduler")


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
