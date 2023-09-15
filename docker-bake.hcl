group "default" {
  targets = ["cpu", "gpu"]
}

target "base" {
  target = "build-stage"
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
}


target "gpu" {
  args = [
    "FROM_IMAGE": "nvidia/cuda:11.8.0-base-ubuntu22.04"
  ]
}

target "gpu" {
  args = [
    "FROM_IMAGE": "ubuntu:22.04"
  ]
}
