import mlx.core as mx


class Flux3Scheduler:
    """Official Flow Map Euler sampling scheduler for FLUX.3 Image Klein."""

    def __init__(self, steps: int = 16, alpha: float = 6.93):
        self.steps = steps
        self.alpha = alpha
        self.schedule = self.compute_schedule(steps, alpha)

    @staticmethod
    def compute_schedule(steps: int = 16, alpha: float = 6.93) -> mx.array:
        t = mx.linspace(1.0, 0.0, steps + 1)
        return alpha * t / (1.0 + (alpha - 1.0) * t)

    def denoise_step(
        self,
        dit,
        packed,
        step_idx: int,
    ) -> None:
        t = self.schedule[step_idx]
        dt = self.schedule[step_idx + 1] - t
        packed.timesteps["image"] = mx.full(
            (packed.data["image"].shape[0],), t, dtype=packed.data["image"].dtype
        )
        target = mx.reshape(self.schedule[step_idx + 1], (1,))
        velocity = dit(packed, timesteps_ref=target)
        packed.data["image"] = packed.data["image"] + dt * velocity

