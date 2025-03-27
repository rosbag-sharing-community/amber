from argparse import ArgumentParser
from amber_mcap.exception import TaskDescriptionError
import os
from yaml import safe_load  # type: ignore
from amber_mcap.automation.detic_image_labeler import DeticImageLabeler
from amber_mcap.automation.clip_image_annotation_filter import ClipImageAnnotationFilter
from amber_mcap.automation.nerf_3d_reconstruction import Nerf3DReconstruction
from amber_mcap.dataset.images_dataset import ImagesDataset, ReadImagesConfig
from amber_mcap.dataset.images_and_annotations_dataset import (
    ImagesAndAnnotationsDataset,
    ReadImagesAndAnnotationsConfig,
)
from amber_mcap.importer.video import VideoImporter, VideoImporterConfig
from amber_mcap.visualization.clip_embeddings_visualization import (
    ClipEmbeddingsVisualization,
)
from typing import Any, Callable, List
import torch


def setup_arguments_and_parser_for_automation(
    parser_automation: Any, func: Callable[[Any], None]
) -> None:
    parser_automation.add_argument(
        "task_description_yaml_path",
        help="Path to the yaml description file path",
        default="",
    )
    parser_automation.add_argument(
        "dataset_description_yaml_path",
        help="Path to the yaml description file path for dataset",
        default="",
    )
    parser_automation.add_argument(
        "rosbag_path",
        help="Path to the target rosbag path",
        default="",
    )
    parser_automation.add_argument(
        "output_rosbag_path",
        help="Path to the output rosbag path",
        default="",
    )
    parser_automation.set_defaults(handler=func)


def setup_arguments_and_parser_for_visualization(
    parser_visualization: Any, func: Callable[[Any], None]
) -> None:
    parser_visualization.add_argument(
        "task_description_yaml_path",
        help="Path to the yaml description file path",
        default="",
    )
    parser_visualization.add_argument(
        "dataset_description_yaml_path",
        help="Path to the yaml description file path for dataset",
        default="",
    )
    parser_visualization.add_argument(
        "rosbag_path",
        help="Path to the target rosbag path",
        default="",
    )
    parser_visualization.set_defaults(handler=func)


def check_config_files_exists_for_automation(args: Any) -> None:
    if not os.path.exists(args.task_description_yaml_path):
        raise TaskDescriptionError(
            "Task description yaml path : "
            + args.task_description_yaml_path
            + " does not exist."
        )
    if not os.path.exists(args.dataset_description_yaml_path):
        raise TaskDescriptionError(
            "Dataset description yaml path : "
            + args.dataset_description_yaml_path
            + " does not exist."
        )
    if not os.path.exists(args.rosbag_path):
        raise TaskDescriptionError(
            "Specified rosbag path : " + args.rosbag_path + " does not exist."
        )


def run_detic_image_labaler_automation(args: Any) -> None:
    check_config_files_exists_for_automation(args)
    labeler = DeticImageLabeler(args.task_description_yaml_path)
    dataset = ImagesDataset(
        args.rosbag_path,
        ReadImagesAndAnnotationsConfig.from_yaml_file(
            args.dataset_description_yaml_path
        ),
    )
    annotations = labeler.inference(dataset)
    labeler.write(
        dataset, "/detic_image_labeler/annotation", annotations, args.output_rosbag_path
    )


def run_clip_image_annotation_filter_automation(args: Any) -> None:
    check_config_files_exists_for_automation(args)
    filter = ClipImageAnnotationFilter(args.task_description_yaml_path)
    dataset = ImagesAndAnnotationsDataset(
        args.rosbag_path,
        ReadImagesAndAnnotationsConfig.from_yaml_file(
            args.dataset_description_yaml_path
        ),
    )
    annotations = filter.inference(dataset)
    filter.write(
        dataset,
        "/clip_image_annotation_filter/annotation",
        annotations,
        args.output_rosbag_path,
    )


def run_nerf_3d_reconstruction_automation(args: Any) -> None:
    check_config_files_exists_for_automation(args)
    task_description = {}
    with open(args.task_description_yaml_path, "rb") as file:
        task_description = safe_load(file)
    reconstruction = Nerf3DReconstruction(args.task_description_yaml_path)
    dataset = ImagesDataset(
        args.rosbag_path,
        ReadImagesConfig.from_yaml_file(args.dataset_description_yaml_path),
    )
    reconstruction.inference(dataset)


def setup_arguments_and_parser_for_import(
    parser_import: Any, func: Callable[[Any], None]
) -> None:
    parser_import.add_argument("video", help="Video file path for importer.")
    parser_import.add_argument("config", help="Config file for video importer.")
    parser_import.set_defaults(handler=func)


def check_config_files_exists_for_import(args: Any) -> None:
    if not os.path.exists(args.config):
        raise TaskDescriptionError(
            "Specified config : " + args.config + " does not exist."
        )


def run_video_import(args: Any) -> None:
    check_config_files_exists_for_import(args)
    VideoImporter(args.video, VideoImporterConfig.from_yaml_file(args.config)).write()


def check_config_files_exists_for_visualize(args: Any) -> None:
    if not os.path.exists(args.task_description_yaml_path):
        raise TaskDescriptionError(
            "Task description yaml path : "
            + args.task_description_yaml_path
            + " does not exist."
        )
    if not os.path.exists(args.dataset_description_yaml_path):
        raise TaskDescriptionError(
            "Specified config : " + args.config + " does not exist."
        )


def run_visualize_image(args: Any) -> None:
    check_config_files_exists_for_visualize(args)
    dataset = ImagesAndAnnotationsDataset(
        args.rosbag_path,
        ReadImagesAndAnnotationsConfig.from_yaml_file(
            args.dataset_description_yaml_path
        ),
    )
    visualization = ClipEmbeddingsVisualization(args.task_description_yaml_path)
    visualization.visualize(dataset)


def main() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    # Setup command line options for automation
    parser_automation = subparsers.add_parser(
        "automation", help="Run automation command"
    )
    subparsers_automation = parser_automation.add_subparsers()
    # Setup command line option for detic image labaler
    parser_detic_image_labeler_automation = subparsers_automation.add_parser(
        "detic_image_labeler", help="Run detic image labaler for rosbag"
    )
    setup_arguments_and_parser_for_automation(
        parser_detic_image_labeler_automation, run_detic_image_labaler_automation
    )
    # Setup command line option for clip image annotation filter
    parser_clip_image_annotation_filter = subparsers_automation.add_parser(
        "clip_image_annotation_filter",
        help="Run clip image annotation filter for rosbag.",
    )
    setup_arguments_and_parser_for_automation(
        parser_clip_image_annotation_filter, run_clip_image_annotation_filter_automation
    )
    # Setup command line option for NeRF 3D reconstruction
    parser_nerf_3d_reconstruction = subparsers_automation.add_parser(
        "nerf", help="Run NeRF 3D reconstruction."
    )
    setup_arguments_and_parser_for_automation(
        parser_nerf_3d_reconstruction, run_nerf_3d_reconstruction_automation
    )

    # Setup command line options for visualization command
    parser_visualization = subparsers.add_parser(
        "visualize", help="Visualize rosbag data."
    )
    # Setup command line options for visualize image_embedding command
    subparsers_visualization = parser_visualization.add_subparsers()
    parser_visualization = subparsers_visualization.add_parser(
        "image_embedding", help="Visualize image embeddings in rosbag data."
    )
    setup_arguments_and_parser_for_visualization(
        parser_visualization, run_visualize_image
    )

    # Setup command line option for import command
    parser_import = subparsers.add_parser(
        "import", help="Import non-rosbag data into rosbag"
    )
    # Setup command line option for import video command
    subparsers_import = parser_import.add_subparsers()
    parser_video_import = subparsers_import.add_parser(
        "video", help="Import video images into rosbag"
    )
    setup_arguments_and_parser_for_import(parser_video_import, run_video_import)

    # Parsing arguments
    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
