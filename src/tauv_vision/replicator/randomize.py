import omni.replicator.core as rep
import sys
import asyncio
import pathlib
import glob
import time
import carb.settings
import uuid

# ~/.local/share/ov/pkg/code-2022.3.3/omni.code.sh --/omni/replicator/script=/home/theo/Documents/yolo_pose/replicator/randomize.py

SCENES_DIR = pathlib.Path("~/Documents/TAUV-Synthetic/scenes").expanduser()
MODELS_DIR = pathlib.Path("~/Documents/TAUV-Synthetic/models").expanduser()
HDRI_DIR = pathlib.Path("~/Documents/TAUV-Synthetic/hdris").expanduser()

dosch_underwater_hdris = glob.glob(str(HDRI_DIR / "dosch_underwater/spherical_map/*XXL.hdr"))
misc_hdris = glob.glob(str(HDRI_DIR / "misc/*.exr"))

objs = [
    str(MODELS_DIR / "new_bin_24/usd/bin_24.usd"),
    # str(MODELS_DIR / "sample_bin_24/usd/sample_bin_24.usd"),
    # str(MODELS_DIR / "buoy_24/usd/buoy_24.usd"),
    # str(MODELS_DIR / "gate_24/usd/gate_24_ccw.usd"),
    # str(MODELS_DIR / "gate_24/usd/gate_24_cw.usd"),
    # str(MODELS_DIR / "path_24/usd/path_24.usd"),
    # str(MODELS_DIR / "samples_24/usd/sample_24_coral.usd"),
    # str(MODELS_DIR / "samples_24/usd/sample_24_nautilus.usd"),
    # str(MODELS_DIR / "samples_24/usd/sample_24_worm.usd"),
    # str(MODELS_DIR / "torpedo_24/usd/torpedo_24.usd"),
]

distractors = [
    str(MODELS_DIR / "angle_distractors/usd/angle_distractors.usd"),
    str(MODELS_DIR / "rope_distractors/usd/rope_distractors.usd"),
]

NUM_FRAMES = 20000

SCENE_PRIM_PREFIX = "/Replicator/Ref_Xform/Ref"

with rep.new_layer():
    scene = rep.create.from_usd(str(SCENES_DIR / "underwater_scene_1/underwater_scene_1.usd"))

    rope_distractors = rep.create.from_usd(str(MODELS_DIR / "rope_distractors/usd/rope_distractors.usd"))
    angle_distractors = rep.create.from_usd(str(MODELS_DIR / "angle_distractors/usd/angle_distractors.usd"))

    with rope_distractors:
        rep.modify.pose(
            scale=(0.1, 0.1, 0.1)
        )

    for obj in objs:
        rep.create.from_usd(obj)

    camera = rep.create.camera(
        position=(0, 0, 0),
        rotation=(0, 0, 0),
    )
    render_product = rep.create.render_product(camera, (640, 360))

    def randomize_sky():
        print("randomizing sky")
        sky = rep.get.prims(f"{SCENE_PRIM_PREFIX}/Environment/sky")

        with sky:
            rep.modify.pose(
                rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            )
            rep.modify.attribute("texture:file", rep.distribution.choice(
                dosch_underwater_hdris + misc_hdris
            ))

            rep.modify.attribute("intensity", rep.distribution.uniform(200, 250))
            rep.modify.attribute("exposure", rep.distribution.uniform(0, 5))

        return sky.node

    rep.randomizer.register(randomize_sky)

    def randomize_sun():
        print("randomizing sun")
        sun = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/Environment/sun")

        with sun:
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, -180, 0), (45, 180, 0)),
            )

            rep.modify.attribute("colorTemperature", rep.distribution.normal(6500, 1000))

            rep.modify.attribute("intensity", rep.distribution.uniform(0, 1000))

        return sun.node

    rep.randomizer.register(randomize_sun)

    def randomize_water():
        print("randomizing water")
        water = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/Looks/Water")

        with water:
            rep.modify.attribute("inputs:volume_scattering", rep.distribution.uniform(0.0, 0.05))
            rep.modify.attribute("inputs:base_thickness", rep.distribution.uniform(1, 5))

        return water.node

    rep.randomizer.register(randomize_water)

    def randomize_environment():
        print("randomizing environment")

        environment = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/Environment")

        with environment:
            rep.modify.pose(
                position=rep.distribution.uniform((0, 200, 0), (0, 1000, 0)),
            )

        return environment.node

    rep.randomizer.register(randomize_environment)

    def randomize_distractors():
        print("getting distractors")
        distractors = rep.get.prims(semantics=[('type', 'distractor')])
        print("got distractors")

        with distractors:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-1, 1),
                vertical_location=rep.distribution.uniform(-1, 1),
                distance=rep.distribution.uniform(800, 1000),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            )

            rep.modify.visibility(
                rep.distribution.choice([True, False], weights=[0.2, 0.8])
            )

            rep.randomizer.color(
                colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
            )

        return distractors.node

    rep.randomizer.register(randomize_distractors)

    def randomize_objects():
        print("randomizing objects")

        obj_prims = rep.get.prims(semantics=[("type", "object")])

        # with obj_prims:
        #     rep.modify.visibility(
        #         rep.distribution.choice([True, False], weights=[0.1, 0.9])
        #     )

        samples = rep.get.prims(semantics=[("class", "sample_24_worm"), ("class", "sample_24_coral"), ("class", "sample_24_nautilus")])

        with samples:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-0.6, 0.6),
                vertical_location=rep.distribution.uniform(-0.6, 0.6),
                distance=rep.distribution.uniform(60, 300),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            )

            rep.randomizer.color(
                colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
            )

        bin_path = rep.get.prims(semantics=[("class", "bin_24"), ("class", "path_24")])

        with bin_path:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-0.6, 0.6),
                vertical_location=rep.distribution.uniform(-0.6, 0.6),
                distance=rep.distribution.uniform(200, 800),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-180, -180, -180), (180, 180, 180)),
            )

        sample_bin = rep.get.prims(semantics=[("class", "sample_bin_24")])

        with sample_bin:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-0.8, 0.8),
                vertical_location=rep.distribution.uniform(-0.8, 0.8),
                distance=rep.distribution.uniform(100, 800),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-60, -60, -180), (60, 60, 180)),
            )

        gate = rep.get.prims(semantics=[("class", "gate_24_ccw"), ("class", "gate_24_cw")])

        with gate:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-0.8, 0.8),
                vertical_location=rep.distribution.uniform(-0.8, 0.8),
                distance=rep.distribution.uniform(100, 600),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-30, -30, -180), (30, 30, 180)),
            )

        buoy = rep.get.prims(semantics=[("class", "buoy_24")])

        with buoy:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-0.8, 0.8),
                vertical_location=rep.distribution.uniform(-0.8, 0.8),
                distance=rep.distribution.uniform(100, 800),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-30, -30, 0), (30, 30, 0)),
            )

        torpedo = rep.get.prims(semantics=[("class", "torpedo_24")])

        with torpedo:
            rep.modify.pose_camera_relative(
                camera = camera,
                render_product = render_product,
                horizontal_location = rep.distribution.uniform(-0.6, 0.6),
                vertical_location = rep.distribution.uniform(-0.6, 0.6),
                distance = rep.distribution.uniform(300, 1000),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-45, -45, -15), (45, 45, -15)),
            )

        return obj_prims.node

    rep.randomizer.register(randomize_objects)

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([render_product])

    bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    bbox_annot.attach([render_product])

    bbox3d_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_3d")
    bbox3d_annot.attach([render_product])

    instance_seg_annot = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast")
    instance_seg_annot.attach([render_product])

    camera_params_annot = rep.AnnotatorRegistry.get_annotator("camera_params")
    camera_params_annot.attach([render_product])

    basic_writer = rep.BasicWriter(
        output_dir=f"/home/theo/Documents/replicator_out/{uuid.uuid4()}",
        colorize_instance_segmentation=False,
    )

    with rep.trigger.on_frame():
        print("trigger")
        rep.randomizer.randomize_sky()
        rep.randomizer.randomize_sun()
        rep.randomizer.randomize_water()
        rep.randomizer.randomize_environment()
        rep.randomizer.randomize_distractors()
        rep.randomizer.randomize_objects()

    async def run():
        await rep.orchestrator.step_async()

        camera_params_data = camera_params_annot.get_data()
        print(f"camera_params_data: {camera_params_data}")

        print("writing camera params!")

        basic_writer.write({
            "trigger_outputs": {"on_time": 0},
            "camera_params": camera_params_data,
        })

        print("wrote camera params")

        rep.settings.set_render_pathtraced(8)

        for i in range(NUM_FRAMES):
            print(f"waiting...")
            sys.stdout.flush()
            await rep.orchestrator.step_async()
            print(f"done waiting")

            rgb_data = rgb_annot.get_data()
            print(f"rgb_data: {rgb_data}")

            bbox_data = bbox_annot.get_data()
            print(f"bbox_data: {bbox_data}")

            bbox3d_data = bbox3d_annot.get_data()
            print(f"fbbox3d_data: {bbox3d_data}")

            instance_seg_data = instance_seg_annot.get_data()
            print(f"instance_seg_data: {instance_seg_data}")

            basic_writer.write({
                "trigger_outputs": {"on_time": 0},
                "rgb": rgb_data,
                "bounding_box_2d_tight": bbox_data,
                "bounding_box_3d": bbox3d_data,
                "instance_segmentation": instance_seg_data,
            })

    asyncio.ensure_future(run())
