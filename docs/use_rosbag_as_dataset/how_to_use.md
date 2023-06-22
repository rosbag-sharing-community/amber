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

If you want to know wahat types of task `amber` supports, please check below.

| Name                                                  | Image              | 2D Annotation      |
|-------------------------------------------------------|--------------------|--------------------|
| [ImagesDataset](../read_images)                       | :heavy_check_mark: |                    |
| [ImagesAndAnnotation](../read_images_and_annotations) | :heavy_check_mark: | :heavy_check_mark: |
