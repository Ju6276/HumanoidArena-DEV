"""Predeclared engineering checks; these do not replace Arena task success."""
import numpy as np
from scipy.spatial.transform import Rotation as R

THRESHOLDS = dict(pose_joint_mae_rad=.15,pose_active_joint_mae_rad=.15,
                  hand_mae_rad=.20,root_height_rmse_m=.12,
                  stationary_root_drift_m=.25,motion_final_position_error_m=.25,
                  motion_yaw_mae_rad=.20,minimum_forward_progress_m=.08)


def probe_metrics(z, case, complete, current):
    n=len(z['q']);ref=z['reference'][:,current];err=z['q']-ref[:,7:]
    values=dict(root_height_rmse_m=float(np.sqrt(np.mean((z['root'][:,2]-ref[:,2])**2))),joint_mae_rad=float(np.mean(np.abs(err))))
    gates=dict(completed_without_original_fall=bool(complete),root_height=values['root_height_rmse_m']<=THRESHOLDS['root_height_rmse_m'])
    if case=='pose':
        names=z['joint_names'].tolist()
        active=[names.index(side+'_'+part+'_joint') for side in ('left','right') for part in ('shoulder_pitch','elbow','wrist_pitch')]
        values['stationary_root_drift_m']=float(np.max(np.linalg.norm(z['root'][:,:2]-z['root'][0,:2],axis=1)))
        gates['stationary_root']=values['stationary_root_drift_m']<=THRESHOLDS['stationary_root_drift_m']
        if n>=600:
            values['active_hold_joint_mae_rad']=float(np.mean(np.abs(err[250:300,active])))
            values['hand_closed_mae_rad']=float(np.mean(np.abs(z['hand_q'][350:400]-z['hand_target'][350:400])))
            values['hand_open_mae_rad']=float(np.mean(np.abs(z['hand_q'][550:600]-z['hand_target'][550:600])))
            gates.update(joint_tracking=values['joint_mae_rad']<=THRESHOLDS['pose_joint_mae_rad'],active_joint_tracking=values['active_hold_joint_mae_rad']<=THRESHOLDS['pose_active_joint_mae_rad'],hands=max(values['hand_closed_mae_rad'],values['hand_open_mae_rad'])<=THRESHOLDS['hand_mae_rad'])
        else:gates['all_segments_exercised']=False
    else:
        values['final_position_error_m']=float(np.linalg.norm(z['root'][-1,:2]-ref[-1,:2]))
        initial=R.from_quat(z['root'][0,3:7][[1,2,3,0]])
        local=initial.inv().apply(z['root'][-1,:3]-z['root'][0,:3]);values['forward_progress_m']=float(local[0])
        if n>=600:
            actual=R.from_quat(z['root'][400:500,3:7][:,[1,2,3,0]]).as_euler('xyz')[:,2]
            target=R.from_quat(ref[400:500,3:7][:,[1,2,3,0]]).as_euler('xyz')[:,2]
            values['yaw_hold_mae_rad']=float(np.mean(np.abs(np.arctan2(np.sin(actual-target),np.cos(actual-target)))))
            gates.update(root_displacement=values['final_position_error_m']<=THRESHOLDS['motion_final_position_error_m'],forward_response=values['forward_progress_m']>=THRESHOLDS['minimum_forward_progress_m'],yaw_tracking=values['yaw_hold_mae_rad']<=THRESHOLDS['motion_yaw_mae_rad'])
        else:gates['all_segments_exercised']=False
    return dict(case=case,thresholds=THRESHOLDS,values=values,gates=gates,passed=all(gates.values()),meaning='engineering controller probe, not Arena task success')
