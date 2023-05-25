# Tips

## I do not have rosbag2, but I have rosbag (ROS1)

[This tool](https://gitlab.com/ternaris/rosbags) can convert your rosbag into rosbag2 very easily.

## My rosbag is not a mcap format.

You can use `ros2 bag convert` command.
First, you prepare compress.yaml like below.

```yaml
output_bags:
  - uri: rosbag
    storage_id: mcap
    compression_mode: message
    compression_format: zstd
    all: true
    compression_queue_size: 0
    compression_threads: 0
```

```bash
ros2 bag convert -i (PATH_TO_ROSBAG) -o conversion.yaml
```
