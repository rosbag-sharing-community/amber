class Schema:
    name: str
    schema_text: str


class ImageMessageSchema(Schema):
    name = "sensor_msgs/Image"
    schema_text = """\
    std_msgs/Header header
    uint32 height
    uint32 width
    string encoding
    uint8 is_bigendian
    uint32 step
    uint8[] data
    ================================================================================
    MSG: std_msgs/Header
    builtin_interfaces/Time stamp
    string frame_id"""
