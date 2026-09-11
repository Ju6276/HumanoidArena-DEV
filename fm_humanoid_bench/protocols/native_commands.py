"""Explicit native-condition track; no task policy and no reference40 claim."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from fm_humanoid_bench.protocols.body_contract import validate_body_output

SCHEMA = 'arena_native_body_v1'


def schema_for(body):
    common = dict(schema=SCHEMA, body=body.name, joint_names=list(body.names), hands='two values in [0,1], shared Dex3', control_dt=.02)
    if body.name=='BFM-Zero':
        common.update(command_fields={'latent':'one declared reward/goal latent name'}, latent_names=sorted(body.latents))
    elif body.name=='SONIC':
        common['command_fields']={'joints':'29 native-order radians or full joint-name map', 'root_quat_wxyz':'unit quaternion in world frame','joint_velocities':'optional 29 rad/s'}
    elif body.name=='TWIST2':
        common['command_fields']={'joints':'29 native-order radians or full joint-name map','velocity':'local forward/left m/s [2]','root_z':'metres','root_roll_pitch':'radians [2]','yaw_rate':'rad/s'}
    else:
        common['command_fields']={'joints':'29 native-order radians or full joint-name map','velocity':'local forward/left m/s [2]','root_z':'metres','root_roll_pitch':'radians [2]','yaw_rate':'rad/s'}
        common['native_mode']='Scale full mode7' if body.name=='ScaleBFM' else 'Holo motion-tracking checkpoint'
    return common


def vector(value, n, label):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f'{label} must be finite [{n}]')
    return a


def validate_command(body, value):
    if not isinstance(value,dict):raise ValueError('Native command must be an object')
    command=copy.deepcopy(value)
    if body.name=='BFM-Zero':
        if set(command)!={'latent'} or command['latent'] not in body.latents:raise ValueError('Unknown native latent')
        return command
    allowed={'joints','root_quat_wxyz','joint_velocities'} if body.name=='SONIC' else {'joints','velocity','root_z','root_roll_pitch','yaw_rate'}
    required=allowed-{'joint_velocities'}
    if set(command)-allowed or required-set(command):raise ValueError('Native command fields do not match Body contract')
    joints=command['joints']
    if isinstance(joints,dict):
        if set(joints)!=set(body.names):raise ValueError('Native command requires all named joints')
        joints=[joints[n] for n in body.names]
    command['joints']=vector(joints,29,'joints').tolist()
    if body.name=='SONIC':
        q=vector(command['root_quat_wxyz'],4,'root quaternion')
        if not np.isclose(np.linalg.norm(q),1,atol=1e-4):raise ValueError('Quaternion must be normalized')
        if 'joint_velocities' in command:command['joint_velocities']=vector(command['joint_velocities'],29,'joint velocities').tolist()
    else:
        for key,n in [('velocity',2),('root_roll_pitch',2)]:command[key]=vector(command[key],n,key).tolist()
        for key in ('root_z','yaw_rate'):
            if isinstance(command[key],bool) or not np.isscalar(command[key]) or not np.isfinite(command[key]):raise ValueError(f'Invalid {key}')
            command[key]=float(command[key])
    return command


def validate_request(body, request):
    allowed={'op','schema','frames','command','hands','prompt_sha256'}
    if set(request)-allowed or request.get('schema')!=SCHEMA or request.get('op')!='native_step':raise ValueError('Invalid native request envelope')
    count=request.get('frames')
    if type(count) is not int or not 1<=count<=250:raise ValueError('frames must be integer 1..250')
    hands=vector(request.get('hands'),2,'hands')
    if np.any((hands<0)|(hands>1)):raise ValueError('Hands must be in [0,1]')
    return validate_command(body,request['command']),count,hands.tolist()


def infer_native(body,state,command):
    command=validate_command(body,command)
    if hasattr(body,'infer_native'):
        return validate_body_output(*body.infer_native(state,command))
    if body.name=='BFM-Zero':
        body.command=command['latent']
        return validate_body_output(*body.infer(state,canonical=False))
    # Existing full-pose native mode accepts a locally specified moving reference.
    # The measured root anchors this native command each tick; it is not reference40.
    quat=np.asarray(state['quat']);r=Rotation.from_quat(quat[[1,2,3,0]])
    yaw=r.as_euler('xyz')[2]
    offsets=np.asarray(body.reference_offsets,dtype=np.float32)*.02
    xy=np.asarray(command['velocity']);v=np.array([np.cos(yaw)*xy[0]-np.sin(yaw)*xy[1],np.sin(yaw)*xy[0]+np.cos(yaw)*xy[1],0])
    pos=np.asarray(state['pos'])[None]+offsets[:,None]*v
    pos[:,2]=command['root_z']
    eulers=np.tile([*command['root_roll_pitch'],yaw],(len(offsets),1));eulers[:,2]+=offsets*command['yaw_rate']
    quats=Rotation.from_euler('xyz',eulers).as_quat()[:,[3,0,1,2]]
    reference=np.c_[pos,quats,np.tile(command['joints'],(len(offsets),1))].astype(np.float32)
    return validate_body_output(*body.infer_reference(state,reference))
