// docker-bake.hcl: Compose images in phases via Buildx contexts

// Build paths (high level):
// - Desktop: base -> common -> desktop_nogpu -> desktop_nogpu_user
// - Jetson Orin: base_orin -> common_orin -> osprey_drivers_orin -> osprey_orin

group "default" {
  targets = ["desktop_nogpu_user"]
}

group "orin" {
  targets = ["osprey_orin"]
}


variable "REGISTRY"     { default = "ghcr.io/tartan-auv" }
variable "IMAGE_NAME"   { default = "desktop_nogpu" }
variable "IMAGE_TAG"    { default = "latest" }

variable "BASE_IMAGE_NAME" { default = "base_nogpu" }

variable "BASE_CONTEXT"      { default = "." }
variable "BASE_DOCKERFILE"   { default = "description/base_nogpu.Dockerfile" }
variable "BASE_ORIN_DOCKERFILE" { default = "description/base_orin.Dockerfile" }
variable "COMMON_DOCKERFILE" { default = "description/common.Dockerfile" }
variable "APP_DOCKERFILE"    { default = "description/desktop.Dockerfile" }
variable "USERCFG_DOCKERFILE" { default = "description/user_config.Dockerfile" }
variable "OSPREY_DRIVERS_DOCKERFILE" { default = "description/osprey_drivers.Dockerfile" }
variable "OSPREY_DEV_DOCKERFILE" { default = "description/osprey_dev.Dockerfile" }

variable "ORIN_IMAGE_NAME" { default = "osprey_orin" }
variable "AWS_CREDENTIALS_FILE" { default = "" }

variable "OPENCV_WITH_CUDA" { default = "OFF" }
variable "OPENCV_CUDA_ARCH_BIN" { default = "" }

// Desktop stack (x86 / host dev)
target "base" {
  context    = "${BASE_CONTEXT}"
  dockerfile = "${BASE_DOCKERFILE}"
  target     = "base"
}

// Jetson Orin stack (aarch64 / JetPack)
target "base_orin" {
  context    = "."
  dockerfile = "${BASE_ORIN_DOCKERFILE}"
  target     = "base"
  secret     = ["id=aws_credentials,src=${AWS_CREDENTIALS_FILE}"]
}

target "common" {
  context    = "."
  dockerfile = "${COMMON_DOCKERFILE}"
  contexts = {
    base = "target:base"
  }
  args = {
    OPENCV_WITH_CUDA     = "${OPENCV_WITH_CUDA}"
    OPENCV_CUDA_ARCH_BIN = "${OPENCV_CUDA_ARCH_BIN}"
  }
}

target "common_orin" {
  context    = "."
  dockerfile = "${COMMON_DOCKERFILE}"
  contexts = {
    base = "target:base_orin"
  }
  args = {
    OPENCV_WITH_CUDA     = "${OPENCV_WITH_CUDA}"
    OPENCV_CUDA_ARCH_BIN = "${OPENCV_CUDA_ARCH_BIN}"
  }
}

// Desktop: common -> desktop image
target "desktop_nogpu" {
  context    = "."
  dockerfile = "${APP_DOCKERFILE}"
  contexts = {
    base = "target:common"
  }
  tags = ["${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"]
}

target "desktop_nogpu_ci" {
  inherits = ["desktop_nogpu"]
  output   = ["type=docker"]
}

target "desktop_nogpu_release" {
  inherits = ["desktop_nogpu"]
  output   = ["type=registry"]
}

target "desktop_nogpu_user" {
  context    = "."
  dockerfile = "${USERCFG_DOCKERFILE}"
  contexts = {
    base = "target:desktop_nogpu"
  }
  tags = ["${REGISTRY}/desktop_nogpu_user:${IMAGE_TAG}"]
  no-cache = true
}

// Orin: common_orin -> osprey_drivers_orin -> osprey_orin
target "osprey_drivers_orin" {
  context    = "."
  dockerfile = "${OSPREY_DRIVERS_DOCKERFILE}"
  contexts = {
    base = "target:common_orin"
  }
  secret = ["id=aws_credentials,src=${AWS_CREDENTIALS_FILE}"]
}

target "osprey_orin" {
  context    = "."
  dockerfile = "${OSPREY_DEV_DOCKERFILE}"
  contexts = {
    base = "target:osprey_drivers_orin"
  }
  tags = ["${REGISTRY}/${ORIN_IMAGE_NAME}:${IMAGE_TAG}"]
}
