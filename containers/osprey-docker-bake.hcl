// docker-bake.hcl: Compose images in phases via Buildx contexts
// Default build: base_nogpu -> common -> desktop_nogpu
// Final image tag: ghcr.io/Tartan-AUV/osprey_orin:latest

// Build the user_config image by default
group "default" {
  targets = ["osprey_orin_user"]
}

// Registry and tagging
variable "REGISTRY"     { default = "ghcr.io/tartan-auv" }
variable "IMAGE_NAME"   { default = "osprey_orin" }
variable "IMAGE_TAG"    { default = "latest" }

// File locations (relative to this HCL file in containers/)
variable "BASE_CONTEXT"      { default = "." }
variable "BASE_DOCKERFILE"   { default = "base_orin/base_orin.Dockerfile" }
variable "COMMON_DOCKERFILE" { default = "common/common.Dockerfile" }
variable "APP_DOCKERFILE"    { default = "osprey_dev/osprey_dev.Dockerfile" }
variable "USERCFG_DOCKERFILE" { default = "user_config/user_config.Dockerfile" }

// Base layer (e.g., Ubuntu + ROS, toolchains, etc.)
// Uses SSH forwarding for git clone in the Dockerfile.
target "base" {
  context    = "${BASE_CONTEXT}"
  dockerfile = "${BASE_DOCKERFILE}"
  target     = "base"
}

// Common layer, built FROM base via BuildKit context named "base"
// Dockerfile: `FROM base as common`
target "common" {
  context    = "."
  dockerfile = "${COMMON_DOCKERFILE}"
  contexts = {
    base = "target:base"
  }
}

// Application layer for osprey_orin, built FROM common via BuildKit context named "base"
// Dockerfile: `FROM base as osprey_orin`
target "osprey_orin" {
  context    = "."
  dockerfile = "${APP_DOCKERFILE}"
  contexts = {
    base = "target:common"
  }
  tags = ["${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"]
}

// CI target: load image into local Docker daemon
target "osprey_orin_ci" {
  inherits = ["osprey_orin"]
  output   = ["type=docker"]
}

// Release target: push image to registry
target "osprey_orin_release" {
  inherits = ["osprey_orin"]
  output   = ["type=registry"]
}

// User-specific config layer applied on top of osprey_orin
// Not cached so it can reflect different users instantly without cache hits
target "osprey_orin_user" {
  context    = "."
  dockerfile = "${USERCFG_DOCKERFILE}"
  contexts = {
    base = "target:osprey_orin"
  }
  tags = ["${REGISTRY}/osprey_orin_user:${IMAGE_TAG}"]
  no-cache = true
  output = ["type=docker"] 
}
