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

| Name                                                  | Image              | 2D Annotation      | PointCloud         |
|-------------------------------------------------------|--------------------|--------------------|--------------------|
| [ImagesDataset](../read_images)                       | :heavy_check_mark: |                    |                    |
| [ImagesAndAnnotation](../read_images_and_annotations) | :heavy_check_mark: | :heavy_check_mark: |                    |
| [PointCLoudsDataset](../read_pointclouds)             |                    |                    | :heavy_check_mark: |

## Sampling data from dataset.

The rosbags record a variety of time series data, but when inputting this data into machine learning, you may want to sample based on conditions rather than inputting everything.  
Amber implements samplers for rosbags to make it easier to perform time series processing.

Currently, only timestamp sampler supports.  
If you want to know details, see [documentation](../timestamp_sampler).
