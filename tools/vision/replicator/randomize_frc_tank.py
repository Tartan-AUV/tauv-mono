import omni.replicator.core as rep
import sys
import asyncio
import carb.settings

# ~/.local/share/ov/pkg/code-2022.3.3/omni.code.sh --/omni/replicator/script=/home/theo/Documents/yolo_pose/replicator/randomize_frc_tank.py

# Camera params fails if this is turned on
NUM_FRAMES = 10000

SCENE_PRIM_PREFIX = "/Replicator/Ref_Xform/Ref"

# carb.settings.get_settings().set("/omni/replicator/RTSubframes", 8)

with rep.new_layer():
    scene = rep.create.from_usd("/home/theo/Documents/yolo_pose/models/frc_tank_scene/frc_tank_scene.usd")

    camera = rep.create.camera(
        position=(0, 800, 0),
        rotation=(-90, 0, 0),
    )
    render_product = rep.create.render_product(camera, (640, 360))

    def randomize_lights():
        print("randomizing lights")
        lights = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/sky")

        with lights:
            rep.modify.attribute("intensity", rep.distribution.uniform(50, 1000))

        return lights.node

    rep.randomizer.register(randomize_lights)

    def randomize_water():
        print("randomizing water")
        water = rep.get.prim_at_path(f"{SCENE_PRIM_PREFIX}/Looks/water")

        with water:
            rep.modify.attribute("inputs:volume_scattering", rep.distribution.uniform(0.01, 0.1))
            rep.modify.attribute("inputs:base_thickness", rep.distribution.uniform(0.1, 0.5))

        return water.node

    rep.randomizer.register(randomize_water)

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

            rep.randomizer.color(
                colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
            )

        return distractors.node

    rep.randomizer.register(randomize_distractors)

    def randomize_objects():
        print("creating objects")
        objects = rep.randomizer.instantiate([
            # "/home/theo/Documents/bin_24/usd/bin_24.usd",
            "/home/theo/Documents/yolo_pose/models/samples_24/usd/worm.usd",
            "/home/theo/Documents/yolo_pose/models/samples_24/usd/coral.usd",
            "/home/theo/Documents/yolo_pose/models/samples_24/usd/nautilus.usd",
        ], size=3, mode="reference", use_cache=True)
        print("created objects")

        with objects:
            rep.modify.pose_camera_relative(
                camera=camera,
                render_product=render_product,
                horizontal_location=rep.distribution.uniform(-0.75, 0.75),
                vertical_location=rep.distribution.uniform(-0.75, 0.75),
                distance=rep.distribution.uniform(100, 300),
            )

            rep.modify.pose(
                rotation=rep.distribution.uniform((-30, -180, -30), (30, 180, 30)),
            )

            rep.randomizer.color(
                colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
            )

        print("done randomizing objects")

        return objects.node

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
        output_dir=f"/home/theo/Documents/new_replicator_out/",
        colorize_instance_segmentation=False,
    )

    with rep.trigger.on_frame():
        print("trigger")
        rep.randomizer.randomize_lights()
        rep.randomizer.randomize_water()
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

        rep.settings.set_render_pathtraced(16)

        # for i in range(NUM_FRAMES):
        #     print(f"waiting...")
        #     sys.stdout.flush()
        #     await rep.orchestrator.step_async()
        #     print(f"done waiting")
        #
        #     rgb_data = rgb_annot.get_data()
        #     print(f"rgb_data: {rgb_data}")
        #
        #     bbox_data = bbox_annot.get_data()
        #     print(f"bbox_data: {bbox_data}")
        #
        #     bbox3d_data = bbox3d_annot.get_data()
        #     print(f"fbbox3d_data: {bbox3d_data}")
        #
        #     instance_seg_data = instance_seg_annot.get_data()
        #     print(f"instance_seg_data: {instance_seg_data}")
        #
        #     # camera_params_data = camera_params_annot.get_data()
        #     # print(f"camera_params_data: {camera_params_data}")
        #
        #     basic_writer.write({
        #         "trigger_outputs": {"on_time": 0},
        #         "rgb": rgb_data,
        #         "bounding_box_2d_tight": bbox_data,
        #         "bounding_box_3d": bbox3d_data,
        #         "instance_segmentation": instance_seg_data,
        #         # "camera_params": camera_params_data,
        #     })

    asyncio.ensure_future(run())
