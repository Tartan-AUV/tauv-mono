// docker-bake.hcl
//
//   docker buildx bake desktop          # build the desktop image into your local Docker
//   docker buildx bake orin             # build the Orin image (arm64) into local Docker
//   docker buildx bake --print desktop  # show the resolved config without building
//
// Release targets (used by CI) push to the registry instead:
//   TAG=$(git rev-parse --short HEAD) docker buildx bake desktop-release orin-release

variable "REGISTRY" { default = "ghcr.io/tartan-auv" }
variable "TAG"      { default = "latest" }

// Container user IDs. The justfile sets these from `id -u` / `id -g`.
variable "HOST_UID" { default = "1000" }
variable "HOST_GID" { default = "1000" }

group "default" {
  targets = ["desktop", "orin"]
}

target "_common" {
  context    = "."
  dockerfile = "Dockerfile"
  args = {
    HOST_UID = HOST_UID
    HOST_GID = HOST_GID
  }
}

target "desktop" {
  inherits = ["_common"]
  target   = "desktop"
  tags     = ["${REGISTRY}/tauv-desktop:${TAG}"]
  output   = ["type=docker"]
}

target "orin" {
  inherits  = ["_common"]
  target    = "orin"
  platforms = ["linux/arm64"]
  args = {
    BASE_IMAGE = "nvcr.io/nvidia/l4t-jetpack:r36.4.0"
    HOST_UID   = HOST_UID
    HOST_GID   = HOST_GID
  }
  tags   = ["${REGISTRY}/tauv-orin:${TAG}"]
  output = ["type=docker"]
}

// ---- CI / release ----

target "desktop-release" {
  inherits   = ["desktop"]
  output     = ["type=registry"]
  cache-from = ["type=registry,ref=${REGISTRY}/tauv-desktop:buildcache"]
  cache-to   = ["type=registry,ref=${REGISTRY}/tauv-desktop:buildcache,mode=max"]
}

target "orin-release" {
  inherits   = ["orin"]
  output     = ["type=registry"]
  cache-from = ["type=registry,ref=${REGISTRY}/tauv-orin:buildcache"]
  cache-to   = ["type=registry,ref=${REGISTRY}/tauv-orin:buildcache,mode=max"]
}
