#!/usr/bin/env python3
"""Train/evaluate residual PPO or review a recorded dynamics rollout."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from robot_core.paths import local_root
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=int,default=0,metavar='STEPS')
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--cycles',type=int,default=3,choices=(1,2,3))
    parser.add_argument('--hand-first-knife',action='store_true',help='Hold knife until fingers reseat, then advance; requires --no-knife-reverse')
    parser.add_argument('--no-knife-reverse',action='store_true')
    parser.add_argument('--continuous-knife',action='store_true')
    parser.add_argument('--regrasp-speed',type=float,default=1.)
    parser.add_argument('--knife-gap-mm',type=float)
    parser.add_argument('--cut-depth-mm',type=float,default=18.)
    parser.add_argument('--support-repeats',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--early-pip-curl',action='store_true')
    parser.add_argument('--landing-wrist-retreat-mm',type=float,default=0.)
    parser.add_argument('--curl-release',action='store_true')
    parser.add_argument('--smooth-regrasp',action='store_true')
    parser.add_argument('--angle-threshold-deg',type=float,default=85.)
    parser.add_argument('--wrist-retreat-mm',type=float,default=0.,help='Wrist retreats with fixed fingertips before angle trigger, 0 to 8 mm')
    parser.add_argument('--wrist-lift-mm',type=float,default=0.,help='Vertical wrist lift during trio regrasp, 0 to 5 mm; orientation fixed')
    parser.add_argument('--pip-guard',action='store_true',help='Experimental arm-assisted PIP constraint; changes wrist motion')
    parser.add_argument('--view',action='store_true')
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--output',type=Path,default=local_root()/'outputs/potato/dynamic_regrasp_restored')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    env=DynamicRegraspEnv(disturbance=False,cycles=args.cycles,pip_guard=args.pip_guard,wrist_lift_m=args.wrist_lift_mm/1000,wrist_retreat_m=args.wrist_retreat_mm/1000,smooth_regrasp=args.smooth_regrasp,angle_threshold_deg=args.angle_threshold_deg,curl_release=args.curl_release,landing_wrist_retreat_m=args.landing_wrist_retreat_mm/1000,early_pip_curl=args.early_pip_curl,support_repeats=args.support_repeats,regrasp_speed=args.regrasp_speed,knife_gap_m=None if args.knife_gap_mm is None else args.knife_gap_mm/1000,cut_depth_m=args.cut_depth_mm/1000,continuous_knife=args.continuous_knife,no_knife_reverse=args.no_knife_reverse,hand_first_knife=args.hand_first_knife)
    policy=None
    if args.train or args.policy:
        from stable_baselines3 import PPO
        if args.train:
            policy=PPO('MultiInputPolicy',env,seed=0,device='cpu',n_steps=512,batch_size=128,
                       learning_rate=3e-4,verbose=1,policy_kwargs=dict(net_arch=[64,64]))
            policy.learn(total_timesteps=args.train)
            policy.save(args.output/'policy')
            (args.output/'training.json').write_text(json.dumps(dict(steps=policy.num_timesteps,seed=0,
                algorithm='PPO',actions='three finger pressure residuals; thumb/little parked; plus shared phase rate',
                wrist_retreat_mm=args.wrist_retreat_mm,wrist_lift_mm=args.wrist_lift_mm,pip_guard=args.pip_guard,hand_gain_multiplier=2.5 if args.pip_guard or args.curl_release else 1.0,hand_first_knife=args.hand_first_knife,no_knife_reverse=args.no_knife_reverse,continuous_knife=args.continuous_knife,regrasp_speed=args.regrasp_speed,knife_gap_mm=args.knife_gap_mm,cut_depth_mm=args.cut_depth_mm,support_repeats=args.support_repeats,early_pip_curl=args.early_pip_curl,landing_wrist_retreat_mm=args.landing_wrist_retreat_mm,curl_release=args.curl_release,smooth_regrasp=args.smooth_regrasp,angle_gate_deg=args.angle_threshold_deg,angle_gate_contact_dwell_s=.06,cycles=args.cycles,contact_shift_m=.024,pressure_direction="world negative Z",external_disturbance=False,bottom_cut_m=.003,shape_randomization=False,privileged_actor=True,knife='close nonpenetrating shallow strokes',
                tactile='ideal MuJoCo contacts; no sensor noise model'),indent=2)+'\n')
        else:policy=PPO.load(args.policy,env=env)
    reports=[];capture=None
    for controller in (['reference','PPO'] if policy else ['reference']):
        for seed in (100,):
            obs,info=env.reset(seed=seed);frames=[];velocities=[];metrics=[];total=0.;pip_starts={}
            while True:
                action=policy.predict(obs,deterministic=True)[0] if controller=='PPO' else np.zeros(4)
                obs,reward,done,truncated,info=env.step(action)
                key=(info['cycle'],info['phase'])
                pip_y=np.asarray(info['pip_positions_m'])[:,1]
                start=pip_starts.setdefault(key,pip_y.copy())
                info['pip_forward_mm']=np.maximum(start-pip_y,0).tolist()
                info['pip_forward_mm']=(np.asarray(info['pip_forward_mm'])*1000).tolist()
                total+=reward;frames.append(env.data.qpos.copy());velocities.append(env.data.qvel.copy());metrics.append(info)
                if done or truncated:break
            reports.append(dict(controller=controller,seed=seed,reward=total,external_disturbance=False,
                policy_source=str(args.policy) if args.policy and controller=="PPO" else None,**info,
                max_pre_retreat_pip_forward_mm=np.max([x['pip_forward_mm'] for x in metrics if x['phase'] in ('CUT_ADVANCE','ANGLE_GATE','KNUCKLE_BACK','TRIO_LIFT','TRIO_RETREAT')],axis=0).tolist(),
                mean_backward_finger_force_n=float(np.mean([np.asarray(x['tip_forces_on_potato_n'])[:,1].sum() for x in metrics])),
                peak_helper_contact_force_n=max(x['max_helper_contact_force_n'] for x in metrics),
                max_knife_hand_force_n=max(x['knife_hand_force_n'] for x in metrics),
                min_knife_hand_gap_m=min(x['knife_hand_gap_m'] for x in metrics),
                min_knife_potato_vertical_clearance_m=min(x['knife_potato_vertical_clearance_m'] for x in metrics)))
            if seed==100:capture=(np.array(frames),np.array(velocities),metrics,controller)
    (args.output/'evaluation.json').write_text(json.dumps(reports,indent=2)+'\n')
    frames,velocities,metrics,label=capture
    np.savez_compressed(args.output/'rollout.npz',qpos=frames,qvel=velocities,dt=.02)
    (args.output/'rollout-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(reports,indent=2),flush=True)
    m=env.model;d=mujoco.MjData(m)
    m.geom_rgba[m.geom("right_blade_visual").id,3]=.3
    m.vis.global_.offwidth=800;m.vis.global_.offheight=600
    camera=mujoco.MjvCamera();camera.lookat[:]=[.433,-.025,.31];camera.distance=.43;camera.azimuth=135;camera.elevation=-25
    def set_frame(i):
        d.qpos[:]=frames[i];d.qvel[:]=velocities[i];d.time=i*.02;mujoco.mj_forward(m,d)
    def mark_middle_bones(scene):
        for finger in (2,3,4):
            a=d.xanchor[m.joint(f'left_finger{finger}_joint3').id]
            b=d.xanchor[m.joint(f'left_finger{finger}_joint4').id]
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),
                               np.eye(3).ravel(),np.array([1.,.2,.1,1.]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0018,a,b)
            scene.ngeom+=1

    def lines(i):
        x=metrics[i]
        knife_status='No external push'
        if env.hand_first_knife:
            if x['phase'] in ('TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE') or (x['phase']=='HOLD_END' and not x['knife_follow_released']):
                knife_status='Knife: WAIT FOR LEFT HAND'
            elif x['phase']=='HOLD_END' and not x['knife_follow_done']:
                knife_status='Knife: FOLLOW LEFT HAND'
            else:knife_status='Knife: READY / GUIDE'
        return [f'CYCLE {x["cycle"]}/{x["total_cycles"]} | {label} | {x["phase"]} | SUPPORT {x["support_repeats_completed"]}/{x["support_repeats"]}',
            f'Potato drift {x["displacement_m"]*1000:.1f} mm | backward finger force {np.asarray(x["tip_forces_on_potato_n"])[:,1].sum():.2f} N',
            'Tip N [thumb index middle ring little]: '+' '.join(f'{v:.2f}' for v in x['tip_loads_n']),
            f'PIP forward from phase start: {max(x["pip_forward_mm"]):.2f} mm | knife gap {x["knife_hand_gap_m"]*1000:.1f} mm',
            'Middle bone deg [index middle ring]: '+', '.join(f'{v:.1f}' for v in x['middle_angles_deg'])+f' | gate {env.angle_threshold_deg:g} deg / 60 ms',
            knife_status+' | gate '+('TRIGGERED' if any(g['cycle']==x['cycle'] for g in x['angle_gate_events']) else 'ARMED')+' | rigid potato']
    if args.export:
        import imageio.v2 as imageio
        from PIL import Image,ImageDraw
        snapshots={}
        for cycle in range(1,args.cycles+1):
            triggered=[i for i,x in enumerate(metrics) if any(g['cycle']==cycle for g in x['angle_gate_events'])]
            if triggered:snapshots[min((triggered[0]+1)//2*2,(len(frames)-1)//2*2)]=f'cycle_{cycle}_trigger.png'
        for cycle in range(1,args.cycles+1):
            completed=[i for i,x in enumerate(metrics) if x['completed_cycles']>=cycle]
            if completed:
                snapshots[min((completed[0]+1)//2*2,(len(frames)-1)//2*2)]=f'cycle_{cycle}_end.png'
        with mujoco.Renderer(m,height=600,width=800) as renderer,imageio.get_writer(str(args.output/'dynamic_regrasp.mp4'),fps=25) as video:
            for i in range(0,len(frames),2):
                set_frame(i);renderer.update_scene(d,camera=camera)
                mark_middle_bones(renderer.scene)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,800,125),fill=(10,20,25))
                for row,line in enumerate(lines(i)):draw.text((8,5+row*20),line,fill='white')
                video.append_data(np.array(frame))
                if i==min(len(frames)-1,300)//2*2:frame.save(args.output/'preview.png')
                if i in snapshots:frame.save(args.output/snapshots[i])
    if args.view:
        from mujoco import viewer
        paused=[False];camera_choice=[None]
        def key(k):
            if k==32:paused[0]=not paused[0]
            if k in (49,50,51):camera_choice[0]={49:(175,-18),50:(90,-85),51:(135,-25)}[k]
        with viewer.launch_passive(m,d,key_callback=key,show_left_ui=False,show_right_ui=False) as window:
            window.cam.lookat[:]=camera.lookat;window.cam.distance=camera.distance
            window.cam.azimuth=camera.azimuth;window.cam.elevation=camera.elevation
            t=0.;last=time.monotonic()
            print('DYNAMIC_VIEWER_READY: SPACE pause; actual physics rollout with contact metrics',flush=True)
            while window.is_running():
                now=time.monotonic()
                if not paused[0]:t+=(now-last)*.5
                last=now;i=min(int(t/.02),len(frames)-1)
                if t>len(frames)*.02+1:t=0.
                with window.lock():
                    set_frame(i)
                    window.user_scn.ngeom=0
                    mark_middle_bones(window.user_scn)
                    if camera_choice[0] is not None:
                        window.cam.azimuth,window.cam.elevation=camera_choice[0]
                        camera_choice[0]=None
                window.set_texts([(100,mujoco.mjtGridPos.mjGRID_TOPLEFT,'\n'.join(lines(i)),'')])
                window.sync();time.sleep(1/60)

if __name__=='__main__':main()
