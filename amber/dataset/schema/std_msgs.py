from amber.dataset.schema import Schema


class Header(Schema):  # type: ignore
    name = "std_msgs/Header"
    schema = """\
    builtin_interfaces/Time stamp
    string frame_id"""
