# AMBER: Annotation and Multimodal Bag Extraction for Robotics

[![pytest](https://github.com/OUXT-Polaris/amber/actions/workflows/pytest.yaml/badge.svg)](https://github.com/OUXT-Polaris/amber/actions/workflows/pytest.yaml)

Amber is a ROS2 friendly ML tools.
Your rosbag2 become dataset!

## Dependencies
### Poetry
Setup environment and dependencies in python.
### Docker
Run some automation algorithums.

## How it works

1. Prepare rosbag2 with mcap format.
2. Prepare task description yaml file.
3. Enjyo your ML life with Robots!

## Limitation
- rosbag2 should be mcap format.
- only zstd message compression supports.

## Task descriptions

You can define which topics should be used.

### Image only

In image_only task, Rosbag2Dataset Class provides only image data.
Example task description is here.

```yaml
dataset_type: image_only
image_topics:
  - topic_name: /wamv/sensors/cameras/front_left_camera_sensor/image_raw
    compressed: true
  - topic_name: /wamv/sensors/cameras/front_right_camera_sensor/image_raw
```

## Tips

### I do not have rosbag2, but I have rosbag (ROS1)

[This tool](https://gitlab.com/ternaris/rosbags) can convert your rosbag into rosbag2 very easily.

### My rosbag is not a mcap format.

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
