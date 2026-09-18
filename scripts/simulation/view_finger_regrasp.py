#!/usr/bin/env python3
"""Review synchronized three-finger regrasp with thumb/little-finger support."""
import argparse
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import mujoco
from mujoco import viewer
import numpy as np

from robot_core.paths import local_root
from twin_sim.finger_regrasp import build_finger_regrasp_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Export a video, phase snapshots and measurements, then exit")
    parser.add_argument("--output", type=Path, default=local_root()/"outputs/potato/finger_regrasp_sync")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = build_finger_regrasp_preview()
    model, data = plan.model, mujoco.MjData(plan.model)
    model.geom_rgba[model.geom("right_blade_visual").id, 3] = .30
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.433, -.025, .31]
    camera.distance = .43
    camera.azimuth, camera.elevation = 135, -25
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0

    def pose(index):
        data.qpos[:] = plan.qpos[index]
        data.qvel[:] = 0
        data.time = plan.times_s[index]
        mujoco.mj_forward(model, data)

    def markers(scene):
        for finger in (2, 3, 4):
            for position, color in ((data.geom_xpos[model.geom(f"left_finger{finger}_pad").id], [0, 1, .3, 1]),
                                    (data.xpos[model.body(f"left_finger{finger}_link3").id]+[0, -.0107, 0], [1, .15, .8, 1])):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.0025]*3), position, np.eye(3).ravel(), np.array(color))
                scene.ngeom += 1
        for finger in (1, 5):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.003]*3),
                data.geom_xpos[model.geom(f"left_finger{finger}_pad").id], np.eye(3).ravel(), np.array([1., .65, .05, 1.]))
            scene.ngeom += 1
        for i in range(12):
            if i == 0:
                a, b = [.384, .085, .232], [.484, .085, .232]
            else:
                x = .384+(i-1)*.01
                a, b = [x, .08, .232], [x, .09, .232]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([1., 1., .3, 1.]))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0007, np.array(a), np.array(b))
            scene.ngeom += 1

    retract = np.array(plan.phases) == "KNUCKLE_BACK"
    report = {
        "kind": "geometric_preview_not_RL_or_force_validation",
        "potato_nominal_length_width_height_mm": [130, 80, 60],
        "hand_scale": 1.0,
        "knuckle": "PIP",
        "sequence": ["PIP retract with anchored tips", "knife clear", "thumb/little support established",
                     "three tips lift together", "three tips retreat together", "three tips re-seat together", "support fingers release"],
        "palm_lowering_mm": plan.palm_lowering_m*1000,
        "trio_lift_mm": 12,
        "trio_retreat_mm": 8,
        "minimum_three_finger_clearance_mm": float(plan.finger_clearances_m.min()*1000),
        "support_signed_distances_during_retreat_mm": (plan.support_distances_m[np.array(plan.phases)=="TRIO_RETREAT"][-1]*1000).tolist(),
        "max_tip_drift_during_knuckle_back_mm": float(np.linalg.norm(plan.tips_m[retract]-plan.tips_m[0], axis=2).max()*1000),
        "initial_PIP_leads_mm": (plan.leads_m[0]*1000).tolist(),
        "retracted_PIP_leads_mm": (plan.leads_m[retract][-1]*1000).tolist(),
        "final_tip_signed_distances_mm": (plan.tip_distances_m[-1]*1000).tolist(),
        "max_left_arm_joint_change_rad": float(np.max(np.abs(plan.qpos[:, plan.left_qpos_ids]-plan.qpos[0, plan.left_qpos_ids]))),
    }
    (args.output/"measurements.json").write_text(json.dumps(report, indent=2)+"\n")

    if args.export:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        snapshots = {0: "01_hold", int(np.where(retract)[0][-1]): "02_knuckle_back"}
        for phase, name in (("SUPPORT_HOLD", "03_support_ready"), ("TRIO_LIFT", "04_trio_lift"),
                            ("TRIO_RETREAT", "05_trio_back"), ("TRIO_PLACE", "06_trio_reseated"), ("HOLD_END", "07_helpers_released")):
            snapshots[int(np.where(np.array(plan.phases) == phase)[0][-1])] = name
        with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(str(args.output/"finger_regrasp.mp4"), fps=25) as video:
            for index in range(len(plan.qpos)):
                if index % 2 and index not in snapshots:
                    continue
                pose(index)
                renderer.update_scene(data, camera=camera, scene_option=option)
                markers(renderer.scene)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 50), fill=(12, 20, 25))
                draw.text((8, 5), f"SYNC TRIO PREVIEW | {plan.phases[index]} | palm -8 mm, FIXED", fill="white")
                draw.text((8, 25), "Green: moving trio | gold: thumb/little supports | yellow ruler: 100 mm", fill="white")
                frame = np.array(image)
                if index % 2 == 0:
                    video.append_data(frame)
                if index in snapshots:
                    imageio.imwrite(args.output/f"{snapshots[index]}.png", frame)
        print(json.dumps(report, indent=2), flush=True)
        return

    keys = SimpleQueue()
    with viewer.launch_passive(model, data, key_callback=keys.put, show_left_ui=False, show_right_ui=False) as window:
        window.cam.lookat[:] = camera.lookat
        window.cam.distance, window.cam.azimuth, window.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        window.opt.sitegroup[:] = 0
        playing, clock, speed, last = True, 0., .5, time.monotonic()
        print("VIEWER_READY: simultaneous three-finger regrasp, thumb/little supports, lower palm; SPACE pause; 1/2/3 cameras; arrows scrub; R reset", flush=True)
        while window.is_running():
            now = time.monotonic()
            elapsed, last = min(now-last, .1), now
            try:
                while True:
                    key = keys.get_nowait()
                    if key == 32:
                        playing = not playing
                    elif key in (82, 114):
                        clock = 0
                    elif key in (262, 263):
                        playing = False
                        clock = float(np.clip(clock+(.04 if key == 262 else -.04), 0, plan.times_s[-1]))
                    elif key in (49, 50, 51):
                        with window.lock():
                            window.cam.azimuth, window.cam.elevation = {49: (175, -18), 50: (90, -85), 51: (135, -25)}[key]
            except Empty:
                pass
            if playing:
                clock += elapsed*speed
                if clock > plan.times_s[-1]+1:
                    clock = 0
            index = min(int(clock/.02), len(plan.qpos)-1)
            with window.lock():
                pose(index)
                window.user_scn.ngeom = 0
                markers(window.user_scn)
            window.set_texts([(100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "SYNCHRONIZED THREE-FINGER REGRASP | GEOMETRIC PREVIEW\n"+plan.phases[index],
                f"{clock:.2f}s | {'PLAYING' if playing else 'PAUSED'}\nPalm -8 mm, FIXED | gold: thumb/little supports"),
                (100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "SPACE play/pause | arrows scrub | 1 side / 2 top / 3 oblique | R restart\nGreen trio moves together | other fingers support | no claim of dynamic stability", "")])
            window.sync()
            time.sleep(1/60)


if __name__ == "__main__":
    main()
