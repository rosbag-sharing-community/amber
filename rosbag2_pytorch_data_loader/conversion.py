from PIL import Image
from mcap.records import Message, Schema
from mcap_ros2.decoder import Decoder
from mcap_ros2._dynamic import DecodedMessage
from pyzstd import decompress


def decompress_message(message: Message) -> Message:
    message.data = decompress(message.data)
    return message


def to_image(ros_message: DecodedMessage) -> Image:
    print(ros_message.encoding)


def decode_image_message(
    message: Message, schema: Schema, decompressed: bool = True
) -> int:
    decoder = Decoder()
    if decompressed:
        ros_message = decoder.decode(schema, decompress_message(message))
    else:
        ros_message = decoder.decode(schema, message)
    to_image(ros_message)
    return 0
