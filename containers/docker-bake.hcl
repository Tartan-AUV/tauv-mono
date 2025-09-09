// docker-bake.hcl: Compose images in phases via Buildx contexts
// Default build: base_nogpu -> common -> desktop_nogpu
// Final image tag: ghcr.io/Tartan-AUV/desktop_nogpu:latest

// Build the desktop_nogpu image by default
group "default" {
  targets = ["desktop_nogpu"]
}

// Registry and tagging
variable "REGISTRY"     { default = "ghcr.io/tartan-auv" }
variable "IMAGE_NAME"   { default = "desktop_nogpu" }
variable "IMAGE_TAG"    { default = "latest" }

// File locations (relative to this HCL file in containers/)
variable "BASE_CONTEXT"      { default = "." }
variable "BASE_DOCKERFILE"   { default = "base_nogpu/base_nogpu.Dockerfile" }
variable "COMMON_DOCKERFILE" { default = "common/common.Dockerfile" }
variable "APP_DOCKERFILE"    { default = "desktop/desktop.Dockerfile" }

// Base layer (e.g., Ubuntu + ROS, toolchains, etc.)
// Uses SSH forwarding for git clone in the Dockerfile.
target "base" {
  context    = "${BASE_CONTEXT}"
  dockerfile = "${BASE_DOCKERFILE}"
  target     = "base"
  ssh        = ["default"]
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

// Application layer for desktop_nogpu, built FROM common via BuildKit context named "base"
// Dockerfile: `FROM base as desktop_nogpu`
target "desktop_nogpu" {
  context    = "."
  dockerfile = "${APP_DOCKERFILE}"
  contexts = {
    base = "target:common"
  }
  tags = ["${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"]
  // Use registry-backed cache to speed up CI builds
  cache-to = ["type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache,mode=max"]
  cache-from = ["type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache"]
}
