// docker-bake.hcl: Compose images in phases via Buildx contexts

group "default" {
  targets = ["desktop_nogpu_user"]
}

variable "REGISTRY"     { default = "ghcr.io/tartan-auv" }
variable "IMAGE_NAME"   { default = "desktop_nogpu" }
variable "IMAGE_TAG"    { default = "latest" }

variable "BASE_CONTEXT"      { default = "." }
variable "BASE_DOCKERFILE"   { default = "base_nogpu/base_nogpu.Dockerfile" }
variable "COMMON_DOCKERFILE" { default = "common/common.Dockerfile" }
variable "APP_DOCKERFILE"    { default = "desktop/desktop.Dockerfile" }
variable "USERCFG_DOCKERFILE" { default = "user_config/user_config.Dockerfile" }

target "base" {
  context    = "${BASE_CONTEXT}"
  dockerfile = "${BASE_DOCKERFILE}"
  target     = "base"
}

target "common" {
  context    = "."
  dockerfile = "${COMMON_DOCKERFILE}"
  contexts = {
    base = "target:base"
  }
}

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
