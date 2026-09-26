import mlx.core as mx

from mflux.models.common.schedulers import register_contrib
from mflux.models.common.schedulers.base_scheduler import BaseScheduler
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler


class ViggleTurboScheduler(BaseScheduler):
    SIGMA_NODES = {
        6: (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
        8: (1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125),
    }

    def __init__(self, config):
        self.config = config
        steps = config.num_inference_steps
        if steps not in self.SIGMA_NODES:
            raise ValueError(
                f"viggle_turbo supports the published 6-step or 8-step schedules, got {steps}. "
                "Use --steps 6 (default) or --steps 8 for dense text."
            )
        self._sigmas = LinearScheduler.shift_sigmas(
            mx.array(self.SIGMA_NODES[steps], dtype=mx.float32), config, use_terminal=False
        )
        self._timesteps = mx.arange(steps, dtype=mx.float32)

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
