from amber.dataset.schema import Schema


class Image(Schema):  # type: ignore
    name = "sensor_msgs/Image"
    schema = """\
    std_msgs/Header header
    uint32 height
    string encoding
    uint8 is_bigendian
    uint32 step
    uint8[] data
    ================================================================================
    MSG: std_msgs/Header
    builtin_interfaces/Time stamp
    string frame_id"""
