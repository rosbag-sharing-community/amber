from PIL import Image
from mcap.records import Message, Schema
from mcap_ros2.decoder import Decoder
from mcap_ros2._dynamic import DecodedMessage
from pyzstd import decompress, compress
from amber.exception import MessageDecodingError
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy
from amber.unit.time import Time, TimeUnit
import math
from sys import byteorder
from typing import Any, List, Callable
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
            raise MessageDecodingError(
                "image_encodings in sensor_msgs/msg/Image is "
                + ros_message.encoding
                + " , it was not supported yet."
            )


def get_pointcloud_field(field_name: str, fields: List[Any]) -> Any:
    for field in fields:
        if field.name == field_name:
            return field
    raise MessageDecodingError(
        "Fields : "
        + field_name
        + " does not existing in fields. Please check rosbag data."
    )


def read_float_value_from_pointcloud_data(
    field_name: str, fields: List[Any], data: np.array
) -> None:
    field = get_pointcloud_field(field_name, fields)

    def get_bytes(data: np.array, offset: int, size: int) -> bytes:
        return bytes(data[offset : offset + size].tobytes())

    # See also https://docs.ros2.org/foxy/api/sensor_msgs/msg/PointField.html
    # INT8
    if field.datatype == 1:
        get_bytes(data, field.offset, 1)
    # UINT8
    elif field.datatype == 2:
        get_bytes(data, field.offset, 1)
    # INT16
    elif field.datatype == 3:
        get_bytes(data, field.offset, 2)
    # UINT16
    elif field.datatype == 4:
        get_bytes(data, field.offset, 2)
    # INT32
    elif field.datatype == 5:
        get_bytes(data, field.offset, 4)
    # UINT32
    elif field.datatype == 6:
        get_bytes(data, field.offset, 4)
    # FLOAT32
    elif field.datatype == 7:
        get_bytes(data, field.offset, 4)
    # FLOAT64
    elif field.datatype == 8:
        get_bytes(data, field.offset, 8)
    else:
        raise MessageDecodingError(
            "Invalid data type : "
            + str(field.datatype)
            + " does not existing in fields. Please check rosbag data."
        )


def ros_message_to_pointcloud(
    ros_message: DecodedMessage,
) -> open3d.geometry.PointCloud:
    pointcloud = open3d.geometry.PointCloud()
    points = numpy.frombuffer(
        ros_message.data, dtype=np.uint8, count=len(ros_message.data), offset=0
    ).reshape(
        [int(len(ros_message.data) / ros_message.point_step), ros_message.point_step]
    )
    for point in points:
        read_float_value_from_pointcloud_data("x", ros_message.fields, point)
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


def decode_pointcloud_message(
    message: Message, schema: Schema, decompress: bool
) -> open3d.geometry.PointCloud:
    return ros_message_to_pointcloud(decode_message(message, schema, decompress))


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
