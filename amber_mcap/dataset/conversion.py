from PIL import Image
from mcap.records import Message, Schema
from mcap_ros2.decoder import Decoder
from mcap_ros2._dynamic import DecodedMessage
from pyzstd import decompress, compress
from amber_mcap.exception import MessageDecodingError
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy
from amber_mcap.unit.time import Time, TimeUnit
import math
from sys import byteorder
from typing import Any, List, Callable
import cv2
import io
import open3d
from struct import unpack
import tf2_amber


def compress_message(message: Message) -> Message:
    message.data = compress(message.data)
    return message


def decompress_message(message: Message) -> Message:
    message.data = decompress(message.data)
    return message


def ros_message_to_image(ros_message: DecodedMessage) -> Image:
    if "encoding" in dir(ros_message):
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
    elif "format" in dir(ros_message):
        match ros_message.format:
            case "rgb8; jpeg compressed bgr8":
                jpeg_data = numpy.frombuffer(
                    ros_message.data,
                    dtype=np.uint8,
                    count=len(ros_message.data),
                    offset=0,
                )
                return Image.fromarray(
                    cv2.cvtColor(
                        cv2.imdecode(jpeg_data, flags=cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB,
                    )
                )
            case _:
                raise MessageDecodingError(
                    "Unsupported compressed message format. Please check rosbag data."
                )
    else:
        raise MessageDecodingError(
            "The image message should have a sensor_msgs/msg/Image or sensor_msgs/msg/CompressedImage type."
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
    field_name: str, fields: List[Any], data: np.array, is_bigendian: bool
) -> float:
    field = get_pointcloud_field(field_name, fields)

    def get_bytes(data: np.array, offset: int, size: int) -> bytes:
        return bytes(data[offset : offset + size].tobytes())

    # See also https://docs.ros2.org/foxy/api/sensor_msgs/msg/PointField.html
    # INT8
    if field.datatype == 1:
        if is_bigendian:
            return float(unpack(">b", get_bytes(data, field.offset, 1))[0])
        else:
            return float(unpack("<b", get_bytes(data, field.offset, 1))[0])
    # UINT8
    elif field.datatype == 2:
        if is_bigendian:
            return float(unpack(">B", get_bytes(data, field.offset, 1))[0])
        else:
            return float(unpack("<B", get_bytes(data, field.offset, 1))[0])
    # INT16
    elif field.datatype == 3:
        if is_bigendian:
            return float(unpack(">h", get_bytes(data, field.offset, 2))[0])
        else:
            return float(unpack("<h", get_bytes(data, field.offset, 2))[0])
    # UINT16
    elif field.datatype == 4:
        if is_bigendian:
            return float(unpack(">H", get_bytes(data, field.offset, 2))[0])
        else:
            return float(unpack("<H", get_bytes(data, field.offset, 2))[0])
    # INT32
    elif field.datatype == 5:
        if is_bigendian:
            return float(unpack(">i", get_bytes(data, field.offset, 4))[0])
        else:
            return float(unpack("<i", get_bytes(data, field.offset, 4))[0])
    # UINT32
    elif field.datatype == 6:
        if is_bigendian:
            return float(unpack(">I", get_bytes(data, field.offset, 4))[0])
        else:
            return float(unpack("<I", get_bytes(data, field.offset, 4))[0])
    # FLOAT32
    elif field.datatype == 7:
        if is_bigendian:
            return float(unpack(">f", get_bytes(data, field.offset, 4))[0])
        else:
            return float(unpack("<f", get_bytes(data, field.offset, 4))[0])
    # FLOAT64
    elif field.datatype == 8:
        if is_bigendian:
            return float(unpack(">d", get_bytes(data, field.offset, 8))[0])
        else:
            return float(unpack("<d", get_bytes(data, field.offset, 8))[0])
    else:
        raise MessageDecodingError(
            "Invalid data type : "
            + str(field.datatype)
            + " does not existing in fields. Please check rosbag data."
        )


# This function was implemented with reference to PointPillars
# (https://github.com/zhulf0804/PointPillars/blob/b9948e73505c8d6bfa631ffdf76c7148e82c5942/utils/io.py#L18-L24)
def ros_message_to_pointcloud(
    ros_message: DecodedMessage,
) -> torch.Tensor:
    pointcloud = open3d.geometry.PointCloud()
    points = numpy.frombuffer(
        ros_message.data, dtype=np.uint8, count=len(ros_message.data), offset=0
    ).reshape(
        [int(len(ros_message.data) / ros_message.point_step), ros_message.point_step]
    )
    tensor = torch.zeros((len(points), 4), dtype=torch.float32)
    for i, point in enumerate(points):
        tensor[i] = torch.Tensor(
            [
                read_float_value_from_pointcloud_data(
                    "x", ros_message.fields, point, ros_message.is_bigendian
                ),
                read_float_value_from_pointcloud_data(
                    "y", ros_message.fields, point, ros_message.is_bigendian
                ),
                read_float_value_from_pointcloud_data(
                    "z", ros_message.fields, point, ros_message.is_bigendian
                ),
                read_float_value_from_pointcloud_data(
                    "intensity", ros_message.fields, point, ros_message.is_bigendian
                ),
            ]
        )
    return tensor


def image_to_tensor(image: Image) -> torch.Tensor:
    return transforms.Compose([transforms.ToTensor()])(image)


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
) -> torch.Tensor:
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


def build_transform_stamped_message(
    message: Message, schema: Schema, decompress: bool
) -> List[tf2_amber.TransformStamped]:
    tf_messages: Any = decode_message(message, schema, decompress)
    tf_amber_messages: List[tf2_amber.TransformStamped] = []
    for message in tf_messages.transforms:
        tf_amber_message = tf2_amber.TransformStamped(
            tf2_amber.Header(
                tf2_amber.Time(message.header.stamp.sec, message.header.stamp.nanosec),
                message.header.frame_id,
            ),
            message.child_frame_id,
            tf2_amber.Transform(
                tf2_amber.Vector3(
                    message.transform.translation.x,
                    message.transform.translation.y,
                    message.transform.translation.z,
                ),
                tf2_amber.Quaternion(
                    message.transform.rotation.x,
                    message.transform.rotation.y,
                    message.transform.rotation.z,
                    message.transform.rotation.w,
                ),
            ),
        )
        tf_amber_messages.append(tf_amber_message)
    return tf_amber_messages


def build_message_from_time(sec: int, nanosec: int):
    return {"sec": sec, "nanosec": nanosec}


def build_message_from_header(stamp: tf2_amber.Time, frame_id: str):
    return {
        "stamp": build_message_from_time(stamp.sec, stamp.nanosec),
        "frame_id": frame_id,
    }


def build_message_from_vector3(x: float, y: float, z: float):
    return {"x": x, "y": y, "z": z}


def build_message_from_quaternion(x: float, y: float, z: float, w: float):
    return {"x": x, "y": y, "z": z, "w": w}


def build_message_from_transform(
    translation: tf2_amber.Vector3, rotation: tf2_amber.Quaternion
):
    return {
        "translation": build_message_from_vector3(
            translation.x, translation.y, translation.z
        ),
        "rotation": build_message_from_quaternion(
            rotation.x, rotation.y, rotation.z, rotation.w
        ),
    }


def build_message_from_transform_stamped(
    header: tf2_amber.Header, child_frame_id: str, transform: tf2_amber.Transform
):
    return {
        "header": build_message_from_header(header.stamp, header.frame_id),
        "child_frame_id": child_frame_id,
        "transform": build_message_from_transform(
            transform.translation, transform.rotation
        ),
    }


def build_message_from_tf(transforms: List[tf2_amber.TransformStamped]):
    ret = {"transforms": []}
    for transform in transforms:
        ret["transforms"].append(
            build_message_from_transform_stamped(
                transform.header, transform.child_frame_id, transform.transform
            )
        )
    return ret
