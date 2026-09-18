"""Coordinated arm/finger IK with explicit PIP fore-aft position targets."""
import mujoco
import numpy as np


class PipGuardSolver:
    def __init__(self, model, data, left, fingers):
        self.model,self.data,self.fingers=model,data,fingers
        self.qids=np.r_[left.qpos_ids,*[f.qids for f in fingers]]
        self.dids=np.r_[left.dof_ids,*[f.dids for f in fingers]]
        self.rest=data.qpos[self.qids].copy()
        self.board=model.geom('chopping_board').id
        self.colliders=[g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith('left_')
                        and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.limits=np.vstack((model.actuator_ctrlrange[left.actuator_ids],*[f.limits for f in fingers]))

    def solve(self, tips, pip_y, tilts=None):
        m,d=self.model,self.data
        def error_jac():
            mujoco.mj_fwdPosition(m,d)
            err=[];rows=[]
            for k,f in enumerate(self.fingers):
                jp=np.zeros((3,m.nv));jt=np.zeros((3,m.nv))
                mujoco.mj_jacGeom(m,d,jt,None,f.tip);mujoco.mj_jacBody(m,d,jp,None,f.pip)
                err.extend(tips[k]-d.geom_xpos[f.tip]);rows.extend(jt[:,self.dids])
                err.append(pip_y[k]-d.xpos[f.pip,1]);rows.append(jp[1,self.dids])
                if tilts is not None:
                    dip=m.body(f'left_finger{f.finger}_link4').id
                    v=d.xpos[dip]-d.xpos[f.pip];jd=np.zeros((3,m.nv))
                    mujoco.mj_jacBody(m,d,jd,None,dip);diff=jd-jp
                    err.append(-4*(v[2]+np.cos(tilts[k])*np.linalg.norm(v)))
                    rows.append(4*(diff[2]+np.cos(tilts[k])*(v@diff)/np.linalg.norm(v))[self.dids])
            for geom in self.colliders:
                line=np.zeros(6)
                distance=mujoco.mj_geomDistance(m,d,geom,self.board,.02,line)
                if distance>=.003:continue
                normal=line[3:]-line[:3]
                normal/=max(np.linalg.norm(normal),1e-12)
                if distance<0:normal*=-1
                jac=np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jac,None,line[:3],int(m.geom_bodyid[geom]))
                err.append(.003-distance);rows.append((-normal@jac)[self.dids])
            return np.asarray(err),np.asarray(rows)
        for _ in range(180):
            err,jac=error_jac()
            if np.max(abs(err))<1e-5:return
            inverse=jac.T@np.linalg.solve(jac@jac.T+1e-8*np.eye(len(err)),np.eye(len(err)))
            delta=inverse@err
            null=np.eye(len(self.qids))-inverse@jac
            delta+=.015*null@(self.rest-d.qpos[self.qids])
            delta*=min(1.,.035/max(abs(delta).max(),1e-12))
            old=d.qpos[self.qids].copy()
            for fraction in (1,.5,.2,.1,.02):
                d.qpos[self.qids]=np.clip(old+fraction*delta,self.limits[:,0],self.limits[:,1])
                trial,_=error_jac()
                if trial@trial<err@err:break
            else:
                d.qpos[self.qids]=old
                break
        err,_=error_jac()
        if np.max(abs(err))>.0001:raise ValueError(f'PIP guard IK residual {np.max(abs(err))*1000:.3f} mm')
