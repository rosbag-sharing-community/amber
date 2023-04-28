from PIL import Image
from mcap.records import Message, Schema
from mcap_ros2.decoder import Decoder
from mcap_ros2._dynamic import DecodedMessage
from pyzstd import decompress
from rosbag2_pytorch_data_loader.exception import ImageDecodingError


def decompress_message(message: Message) -> Message:
    message.data = decompress(message.data)
    return message


def to_image(ros_message: DecodedMessage) -> Image:
    match ros_message.encoding:
        case "rgb8":
            return Image.frombytes(
                "RGB", (ros_message.width, ros_message.height), ros_message.data
            )
        case _:
            raise ImageDecodingError(
                "image_encodings in sensor_msgs/msg/Image is "
                + ros_message.encoding
                + " , it was not supported yet."
            )


def decode_image_message(
    message: Message, schema: Schema, decompressed: bool = True
) -> Image:
    decoder = Decoder()
    if decompressed:
        ros_message = decoder.decode(schema, decompress_message(message))
    else:
        ros_message = decoder.decode(schema, message)
    return to_image(ros_message)
