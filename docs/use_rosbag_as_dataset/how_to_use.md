# How to use?

## Prepare rosbag data

!!! warning
    Currently, some limitations are exists.
    rosbag2 should be mcap format.
    only zstd message compression supports.

## Use rosbag as dataset

Rosbag includes multimodal data, such as image/pointcloud/audio/etc...
So, you need to describe what data you want to extract from rosbag.
In order to specify this, write yaml setting file.


```python
dataset = ImagesDataset("(path to rosbag .mcap file)", "(path to rosbag yaml description file)")
```

If you want to know wahat types of task `amber` supports, please check task description section in this document.
