from argparse import ArgumentParser
from amber.exception import TaskDescriptionError
import os
from yaml import safe_load  # type: ignore
from amber.automation.detic_image_labeler import DeticImageLabeler
from amber.automation.task_description import AutomationTaskType
from amber.dataset.rosbag2_dataset import Rosbag2Dataset
from typing import Any


def automation(args: Any) -> None:
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
    task_description = {}
    with open(args.task_description_yaml_path, "rb") as file:
        task_description = safe_load(file)
    if task_description["task_type"] == AutomationTaskType.DETIC_IMAGE_LABELER.value:
        labeler = DeticImageLabeler(args.task_description_yaml_path)
        dataset = Rosbag2Dataset(args.rosbag_path, args.dataset_description_yaml_path)
        annotations = labeler.inference(dataset)
        labeler.write(
            dataset, "/detic_image_labeler", annotations, args.output_rosbag_path
        )
        return
    raise TaskDescriptionError(
        "Task description type is invalid, please check task description.yaml. task_type you specified is "
        + task_description["task_type"]
    )


def main() -> None:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    parser_automation = subparsers.add_parser(
        "automation", help="Run automation command"
    )
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
    parser_automation.set_defaults(handler=automation)
    args = parser.parse_args()
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
