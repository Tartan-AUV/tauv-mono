import asyncio
import omni.replicator.core as rep

SCENE_PRIM_PREFIX = "/Replicator/Ref_Xform/Ref"

rep.settings.set_render_pathtraced(16)

with rep.new_layer():
    # scene = rep.create.from_usd("/home/theo/Documents/yolo_pose/models/underwater_scene_1/underwater_scene_1.usd")

    # cam1 = rep.create.camera(f_stop=1.8)
    cam1 = rep.create.camera(
        position=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    rp1 = rep.create.render_product(cam1, (640, 360))

    cam_params_annot1 = rep.AnnotatorRegistry.get_annotator("camera_params")
    cam_params_annot1.attach([rp1])

    basic_writer = rep.BasicWriter(
        output_dir=f"/home/theo/Documents/replicator_out/",
        colorize_instance_segmentation=False,
        # camera_params=True,
    )

    NUM_FRAMES = 10

    async def get_camera_params_async():
        # NOTE: step_async() is needed to feed the annotator with new data
        for i in range(NUM_FRAMES):
            await rep.orchestrator.step_async()
            await asyncio.sleep(1)
            data1 = cam_params_annot1.get_data()
            print(f"data1={data1}")
            await asyncio.sleep(1)

    task = asyncio.ensure_future(get_camera_params_async())