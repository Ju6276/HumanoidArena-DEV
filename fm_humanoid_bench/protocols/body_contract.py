"""Explicit semantics at the reference / learned-controller boundary.

Acceptance means a command can be encoded, not that a learned policy will track
it. Requirements let a caller reject approximations before physics advances.
"""
import numpy as np


CAPABILITIES = {
    'SONIC': {'joint_pose':'native joint29 encoder condition', 'root_tilt':'target relative to measured root orientation', 'root_heading_error':'target relative to measured root orientation', 'hands':'shared Dex3 outside BFM'},
    'TWIST2': {'joint_pose':'native mimic35 joint references', 'root_velocity':'reference-base-local XY velocity and yaw rate', 'root_height':'mimic35 height', 'root_tilt':'mimic35 roll/pitch', 'hands':'shared Dex3 outside BFM'},
    'ScaleBFM': {
        'joint_pose': 'FK of 29 joints, 14 selected links, full mode 7',
        'root_velocity': 'future reference positions',
        'root_height': 'relative target link positions',
        'root_tilt': 'relative target link orientations',
        'root_position_error': 'target minus measured root position',
        'root_heading_error': 'target relative to measured root orientation',
        'hands': 'shared Dex3 position controller, outside the BFM',
    },
    'HoloMotion': {
        'joint_pose': 'current and future 29 joint references',
        'root_velocity': 'finite-difference reference local velocity',
        'root_height': 'reference height only',
        'root_tilt': 'reference gravity and relative future orientations',
        'root_heading_error': 'reference versus measured yaw',
        'hands': 'shared Dex3 position controller, outside the BFM',
    },
    'BFM-Zero': {
        'joint_pose': 'FK pose and velocity through released backward encoder',
        'root_velocity': 'reference body velocities through backward encoder',
        'root_height': 'reference height through backward encoder',
        'root_tilt': 'reference gravity and heading-local body orientations',
        'hands': 'shared Dex3 position controller, outside the BFM',
    },
}

LOSSES = {
    'SONIC': ['selected HA joint29 mode does not condition root XY displacement or reference height'],
    'TWIST2': ['absolute root XY position error is not conditioned', 'absolute heading error is not conditioned; yaw motion is velocity-conditioned'],
    'ScaleBFM': [],
    'HoloMotion': ['absolute root XY position error is not conditioned'],
    'BFM-Zero': ['absolute root XY position error is not conditioned',
                 'absolute heading error is not conditioned; yaw motion is velocity-conditioned'],
}


def describe_body(name, offsets):
    return dict(schema='arena_body_contract_v1', body=name, control_dt=.02,
                reference_offsets=np.asarray(offsets).tolist(),
                capabilities=CAPABILITIES[name], semantic_losses=LOSSES[name],
                acceptance_is_tracking_success=False)


def accept_reference(name, offsets, required=(), allow_approximation=False):
    contract = describe_body(name, offsets)
    if type(allow_approximation) is not bool:
        raise ValueError('allow_native_approximation must be boolean')
    if not isinstance(required, (list, tuple)) or not all(isinstance(k, str) for k in required):
        raise ValueError('required_capabilities must be a list of names')
    missing = sorted(set(required)-CAPABILITIES[name].keys())
    if missing:
        raise ValueError(f'{name} cannot provide required capabilities: {missing}')
    if LOSSES[name] and not allow_approximation:
        raise ValueError(f'{name} requires explicit allow_native_approximation: {LOSSES[name]}')
    return dict(accepted=True, required_capabilities=list(required),
                allow_native_approximation=bool(allow_approximation), **contract)


def validate_body_input(state, reference, names, offsets):
    n = len(names)
    if n != 29 or len(set(names)) != 29:
        raise ValueError('Body must declare 29 unique G1 joints')
    for key, shape in [('q',(n,)),('dq',(n,)),('quat',(4,)),('omega',(3,)),('pos',(3,))]:
        value = np.asarray(state[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid measured {key}: expected {shape}')
    a = np.asarray(reference)
    if a.shape != (len(offsets),36) or not np.isfinite(a).all():
        raise ValueError(f'Expected finite native-order reference [{len(offsets)},36]')
    if not np.isclose(np.linalg.norm(state['quat']),1.,atol=1e-4):
        raise ValueError('Measured quaternion must be normalized wxyz')
    if not np.allclose(np.linalg.norm(a[:,3:7],axis=1),1.,atol=1e-4):
        raise ValueError('Reference quaternions must be normalized wxyz')


def validate_body_output(target, data):
    target = np.asarray(target)
    if target.shape != (29,) or not np.isfinite(target).all():
        raise ValueError('Learned Body must return 29 finite position targets')
    for key in ('obs','raw_action'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Nonfinite native {key}')
    return target, data


def validate_reference_request(request, name, offsets):
    """Validate completely before the simulator changes any command state."""
    from fm_humanoid_bench.protocols.reference import SCHEMA, validate_chunk
    allowed={'op','schema','control_dt','action_chunk','frames','brain_protocol',
             'prediction_step','required_capabilities','allow_native_approximation'}
    if set(request)-allowed:
        raise ValueError(f'Unknown reference request fields: {sorted(set(request)-allowed)}')
    if request.get('schema')!=SCHEMA or request.get('control_dt')!=.02:
        raise ValueError('Reference schema/dt mismatch')
    count=request.get('frames',50)
    if type(count) is not int or not 1<=count<=250:
        raise ValueError('frames must be an integer in 1..250')
    receipt=accept_reference(name,offsets,request.get('required_capabilities',[]),request.get('allow_native_approximation',False))
    actions=validate_chunk(request['action_chunk'])
    if count>len(actions):raise ValueError('Execute prefix exceeds prediction')
    return actions,count,receipt
