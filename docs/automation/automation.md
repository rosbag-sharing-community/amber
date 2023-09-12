# Automation

Automation helps you to enjoy your ML life.
Automation tools can be used with two ways, use with CLI and use with Python API

## CLI tools

All of the automation tools can be use `amber automation` command.
If you want to check the help text, please type `amber automation --help`

## Python API

All of the automation tools have Python classes.
If you want to know detail, please read detatiled documentations.

## Support status

| Name                                                         | Docker Support (CPU) | Docker Support(CUDA) | Native Support     | CUDA Support(Native) | Huggingface Support |
|--------------------------------------------------------------|----------------------|----------------------|--------------------|----------------------|---------------------|
| [DeticImageLabaler](../detic_image_labaler)                  |                      |                      | :heavy_check_mark: |                      |                     |
| [ClipImageAnnotationFilter](../clip_image_annotation_filter) |                      |                      | :heavy_check_mark: |                      |                     |
| [NeRF 3D Reconstruction](../nerf_3d_reconstruction)          |                      | :heavy_check_mark:   |                    |                      |                     |

### Docker Support
Support automation algorithm inside docker.

### Native Support
Support automation algorithm in native environment.

### Hugging face Support
Support running automation algorithum on [hugging face.](https://huggingface.co/)

### CUDA Support
Support cuda for accelerating automation.
