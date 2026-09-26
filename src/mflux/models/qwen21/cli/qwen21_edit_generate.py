import argparse
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.qwen21.qwen21_lora import Qwen21LoRA
from mflux.models.qwen21.reference import QwenImage21Edit
from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil
from mflux.utils.scale_factor import ScaleFactor

IGNORED_OPTIONS = {"--lora-style": "Named LoRA styles apply to Flux in-context only; use --lora."}

CONDITIONAL_OPTIONS = {
    "--scheduler": {
        "condition": "linear for base, viggle_turbo with the Turbo preset",
        "reason": "Incompatible schedulers exit with an error before model loading.",
    },
    "--negative-prompt": {
        "condition": "guidance greater than 1",
        "reason": "Guidance 1 runs only the positive branch.",
    },
    "--independent-edit-noise": {
        "condition": "one or more reference images",
        "reason": "Mixes the seed with reference pixels; text-only generation keeps its original noise.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate and edit RGB/RGBA images with Qwen-Image-2.1.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model="qwen-image-2.1")
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.set_defaults(width=None, height=None, scheduler=None)
    parser.lora_preset_default_steps = {Qwen21LoRA.PRESET: Qwen21LoRA.DEFAULT_STEPS}
    parser._option_string_actions["--scheduler"].help = "Automatic: linear for base, viggle_turbo for Turbo."
    for flag in ("--width", "--height"):
        parser._option_string_actions[flag].help = (
            "Output size: a multiple of 32 or a reference-image scale factor (auto means 1x). "
            "If omitted, derive from output-resolution and the last reference's aspect ratio."
        )
    parser.add_image_paths_arguments(required=False)
    parser.add_output_arguments()
    parser.add_lora_arguments(bake_lora_default=False)
    parser.add_argument(
        "--lora-preset",
        choices=[Qwen21LoRA.PRESET],
        help="Use the Viggle v0.2.1 runtime adapter: 6 steps by default, or 8 for dense text.",
    )
    parser.add_argument(
        "--output-resolution",
        type=int,
        default=1024,
        help="Pixel-area budget for automatic output dimensions (default: 1024).",
    )
    parser.add_argument(
        "--reference-resolution",
        type=int,
        default=1024,
        help="Independent reference-image encoding budget (default: 1024, as in Turbo training).",
    )
    parser.add_argument(
        "--use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse text and reference image prefix attention across steps (default: on).",
    )
    parser.add_argument(
        "--independent-edit-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive reproducible edit noise from the seed and reference pixels to avoid self-edit artifacts (default: on).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    if args.guidance is None or args.guidance == 1:
        CommandLineParser.warn_ignored_options(
            {"--negative-prompt": CONDITIONAL_OPTIONS["--negative-prompt"]["reason"]}
        )
    if Path(args.output).suffix.lower() not in (".png", ".webp", ".tif", ".tiff"):
        parser.error("Qwen-Image-2.1 outputs RGBA; use PNG, WebP or TIFF to retain transparency.")
    paths = args.image_paths or []
    if len(paths) > 10:
        parser.error("Qwen-Image-2.1 supports at most 10 reference images.")
    try:
        args.scheduler = Qwen21LoRA.resolve_scheduler(args.lora_preset, args.scheduler)
        if args.scheduler not in ("linear", "viggle_turbo"):
            raise ValueError("Qwen-Image-2.1 editing supports linear or viggle_turbo Euler sampling only.")
        Qwen21LoRA.validate_generation(
            args.lora_preset, args.steps, 1.0 if args.guidance is None else args.guidance, args.negative_prompt
        )
        if args.lora_preset and args.bake_lora:
            raise ValueError("viggle-turbo requires --no-bake-lora (the Qwen 2.1 default).")
        width, height = args.width, args.height
        if isinstance(width, ScaleFactor) or isinstance(height, ScaleFactor):
            width, height = DimensionResolver.resolve(
                width=width if width is not None else ScaleFactor(1),
                height=height if height is not None else ScaleFactor(1),
                reference_image_path=paths[-1] if paths else None,
            )
        QwenImage21LatentCreator.validate(
            width if width is not None else 32, height if height is not None else 32, args.steps, len(paths)
        )
        QwenImage21LatentCreator.validate_resolution(args.output_resolution)
        QwenImage21LatentCreator.validate_resolution(args.reference_resolution)
        model_config = ConfigResolution.resolve_restricted(
            args.model, "qwen-image-2.1", model_path=args.model_path, base_model=args.base_model
        )
    except (ModelConfigError, ValueError) as exc:
        parser.error(str(exc))
    model = QwenImage21Edit(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
        lora_preset=args.lora_preset,
        bake_lora=args.bake_lora,
    )
    memory_saver = CallbackManager.register_callbacks(args, model, QwenImage21LatentCreator)
    try:
        for seed in args.seed:
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=PromptUtil.read_negative_prompt(args),
                width=width,
                height=height,
                num_inference_steps=args.steps,
                guidance=args.guidance if args.guidance is not None else 1.0,
                image_paths=paths,
                output_resolution=args.output_resolution,
                reference_resolution=args.reference_resolution,
                use_kv_cache=args.use_kv_cache,
                scheduler=args.scheduler,
                independent_edit_noise=args.independent_edit_noise,
            )
            image.save(Path(args.output.format(seed=seed)), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
