class Schema:
    name: str
    schema_text: str


class StringMessageSchema(Schema):
    name = "std_msgs/msg/String"
    schema_text = """\
string data"""


class ImageMessageSchema(Schema):
    name = "sensor_msgs/msg/Image"
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


class CompressedMessageSchema(Schema):
    name = "sensor_msgs/msg/CompressedImage"
    schema_text = """\
std_msgs/Header header
string format
uint8[] data
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id"""


class TFMessageSchema(Schema):
    name = "tf2_msgs/msg/TFMessage"
    schema_text = """geometry_msgs/TransformStamped[] transforms\n\n================================================================================\nMSG: geometry_msgs/TransformStamped\n# This expresses a transform from coordinate frame header.frame_id\n# to the coordinate frame child_frame_id at the time of header.stamp\n#\n# This message is mostly used by the\n# <a href="https://index.ros.org/p/tf2/">tf2</a> package.\n# See its documentation for more information.\n#\n# The child_frame_id is necessary in addition to the frame_id\n# in the Header to communicate the full reference for the transform\n# in a self contained message.\n\n# The frame id in the header is used as the reference frame of this transform.\nstd_msgs/Header header\n\n# The frame id of the child frame to which this transform points.\nstring child_frame_id\n\n# Translation and rotation in 3-dimensions of child_frame_id from header.frame_id.\nTransform transform\n\n================================================================================\nMSG: geometry_msgs/Transform\n# This represents the transform between two coordinate frames in free space.\n\nVector3 translation\nQuaternion rotation\n\n================================================================================\nMSG: geometry_msgs/Quaternion\n# This represents an orientation in free space in quaternion form.\n\nfloat64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1\n\n================================================================================\nMSG: geometry_msgs/Vector3\n# This represents a vector in free space.\n\n# This is semantically different than a point.\n# A vector is always anchored at the origin.\n# When a transform is applied to a vector, only the rotational component is applied.\n\nfloat64 x\nfloat64 y\nfloat64 z\n\n================================================================================\nMSG: std_msgs/Header\n# Standard metadata for higher-level stamped data types.\n# This is generally used to communicate timestamped data\n# in a particular coordinate frame.\n\n# Two-integer timestamp that is expressed as seconds and nanoseconds.\nbuiltin_interfaces/Time stamp\n\n# Transform frame with which this data is associated.\nstring frame_id\n\n================================================================================\nMSG: builtin_interfaces/Time\n# This message communicates ROS Time defined here:\n# https://design.ros2.org/articles/clock_and_time.html\n\n# The seconds component, valid over all int32 values.\nint32 sec\n\n# The nanoseconds component, valid in the range [0, 10e9).\nuint32 nanosec\n"""
    encoding = "ros2msg"
