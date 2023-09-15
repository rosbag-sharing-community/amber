nvidia/cuda:11.8.0-base-ubuntu22.04

group "default" {
  targets = ["cpu", "gpu"]
}
