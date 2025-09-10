// docker-bake.hcl: Compose images in phases via Buildx contexts
// Default build: base_nogpu -> common -> desktop_nogpu
// Final image tag: ghcr.io/Tartan-AUV/desktop_nogpu:latest

// Build the user_config image by default
group "default" {
  targets = ["desktop_nogpu_user"]
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
variable "USERCFG_DOCKERFILE" { default = "user_config/user_config.Dockerfile" }

// Base layer (e.g., Ubuntu + ROS, toolchains, etc.)
// Uses SSH forwarding for git clone in the Dockerfile.
target "base" {
  context    = "${BASE_CONTEXT}"
  dockerfile = "${BASE_DOCKERFILE}"
  target     = "base"
  // Enable both local and registry-backed caches for local and CI builds
  cache-to = [
    "type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache,mode=max",
  ]
  cache-from = [
    "type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache",
  ]
}

// Common layer, built FROM base via BuildKit context named "base"
// Dockerfile: `FROM base as common`
target "common" {
  context    = "."
  dockerfile = "${COMMON_DOCKERFILE}"
  contexts = {
    base = "target:base"
  }
  // Use the same cache settings to ensure cross-target reuse
  cache-to = [
    "type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache,mode=max",
  ]
  cache-from = [
    "type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache",
  ]
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
  cache-to = [
    "type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache,mode=max",
  ]
  cache-from = [
    "type=registry,ref=ghcr.io/tartan-auv/desktop_nogpu:buildcache",
  ]
}

// CI target: load image into local Docker daemon
target "desktop_nogpu_ci" {
  inherits = ["desktop_nogpu"]
  output   = ["type=docker"]
}

// Release target: push image to registry
target "desktop_nogpu_release" {
  inherits = ["desktop_nogpu"]
  output   = ["type=registry"]
}

// User-specific config layer applied on top of desktop_nogpu
// Not cached so it can reflect different users instantly without cache hits
target "desktop_nogpu_user" {
  context    = "."
  dockerfile = "${USERCFG_DOCKERFILE}"
  contexts = {
    base = "target:desktop_nogpu"
  }
  tags = ["${REGISTRY}/desktop_nogpu_user:${IMAGE_TAG}"]
  no-cache = true
}
