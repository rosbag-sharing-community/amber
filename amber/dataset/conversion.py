from PIL import Image
from mcap.records import Message, Schema
from mcap_ros2.decoder import Decoder
from mcap_ros2._dynamic import DecodedMessage
from pyzstd import decompress, compress
from amber.exception import ImageDecodingError
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy
from amber.unit.time import Time, TimeUnit
import math
from sys import byteorder
from typing import Any
import cv2
import io
import open3d


def compress_message(message: Message) -> Message:
    message.data = compress(message.data)
    return message


def decompress_message(message: Message) -> Message:
    message.data = decompress(message.data)
    return message


def ros_message_to_image(ros_message: DecodedMessage) -> Image:
    match ros_message.encoding:
        case "rgb8":
            return Image.frombytes(
                "RGB", (ros_message.width, ros_message.height), ros_message.data
            )
        case "8UC3":
            image = Image.frombytes(
                "RGB", (ros_message.width, ros_message.height), ros_message.data
            )
            b, g, r = image.split()
            return Image.merge("RGB", (r, g, b))
        case _:
            raise ImageDecodingError(
                "image_encodings in sensor_msgs/msg/Image is "
                + ros_message.encoding
                + " , it was not supported yet."
            )


def ros_message_to_pointcloud(
    ros_message: DecodedMessage,
) -> open3d.geometry.PointCloud:
    pointcloud = open3d.geometry.PointCloud()
    return pointcloud


def image_to_tensor(image: Image) -> torch.Tensor:
    return transforms.Compose([transforms.PILToTensor()])(image)


def decode_message(message: Message, schema: Schema, decompress: bool) -> Message:
    decoder = Decoder()
    if decompress:
        return decoder.decode(schema, decompress_message(message))
    else:
        return decoder.decode(schema, message)


def decode_image_message(
    message: Message, schema: Schema, decompress: bool
) -> torch.Tensor:
    return image_to_tensor(
        ros_message_to_image(decode_message(message, schema, decompress))
    )


def build_message_from_image(
    image: numpy.ndarray, frame_id: str, stamp: Time, image_encodings: str = "bgr8"
) -> Any:
    data = image.flatten()
    floored_stamp: float = math.floor(stamp.get(TimeUnit.SECOND))
    return {
        "header": {
            "stamp": {
                "sec": int(floored_stamp),
                "nanosec": Time(
                    stamp.get(TimeUnit.SECOND) - floored_stamp, TimeUnit.SECOND
                ).get(TimeUnit.NANOSECOND),
            },
            "frame_id": frame_id,
        },
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "encoding": image_encodings,
        "is_bigendian": True if byteorder == "big" else False,
        "step": len(data) // int(image.shape[0]),
        "data": data.tolist(),
    }
