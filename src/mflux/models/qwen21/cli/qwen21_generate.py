from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.qwen21_lora import Qwen21LoRA
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "qwen-image-2.1"
IGNORED_OPTIONS = {"--lora-style": "Named LoRA styles apply to Flux in-context only; use --lora."}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image 2.1 model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    parser.add_lora_arguments(bake_lora_default=False)
    parser.add_step_cache_arguments()
    parser.set_defaults(scheduler=None)
    parser.lora_preset_default_steps = dict(Qwen21LoRA.PRESET_DEFAULT_STEPS)
    parser._option_string_actions["--scheduler"].help = "Automatic: linear for base, viggle_turbo for Turbo."
    parser.add_argument(
        "--lora-preset",
        choices=list(Qwen21LoRA.PRESETS),
        help="Acceleration LoRA preset: 'viggle-turbo' (6/8 steps, viggle_turbo) or 'turbo8' (8 steps, linear).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    try:
        args.scheduler = Qwen21LoRA.resolve_scheduler(args.lora_preset, args.scheduler)
        Qwen21LoRA.validate_generation(
            args.lora_preset, args.steps, 1.0 if args.guidance is None else args.guidance, args.negative_prompt
        )
        if args.lora_preset and args.bake_lora:
            raise ValueError("viggle-turbo requires --no-bake-lora (the Qwen 2.1 default).")
        if args.lora_preset and args.image_path is not None:
            raise ValueError("Use mflux-generate-qwen-2.1-edit --image-paths for Turbo reference editing.")
    except ValueError as exc:
        parser.error(str(exc))

    model_config = ConfigResolution.resolve_restricted(
        args.model,
        DEFAULT_MODEL,
        model_path=args.model_path,
        base_model=args.base_model,
        extra_keys=("qwen-image-2.1-heretic-8bit", "qwen-image-2.1-heretic-4bit", "qwen-image-2.1-heretic-bf16"),
    )

    qwen = QwenImage21(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        lora_preset=args.lora_preset,
        bake_lora=args.bake_lora,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=Qwen21LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        for seed in args.seed:
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=width,
                height=height,
                guidance=args.guidance if args.guidance is not None else 1.0,
                scheduler=args.scheduler,
                image_path=args.image_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                step_cache_ratio=args.step_cache_ratio,
            )
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
