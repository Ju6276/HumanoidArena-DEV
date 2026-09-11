"""Verify native Sonic multicam recording against actual physics step ledger."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def verify(directory):
    directory = Path(directory).resolve()
    episodes = sorted({p.resolve() for p in directory.glob('*.npz')})
    if len(episodes) != 1:
        raise ValueError(f'Expected exactly one native episode, found {len(episodes)}')
    rows = [json.loads(x) for x in (directory/'simulation_timeline.jsonl').read_text().splitlines()]
    with np.load(episodes[0], allow_pickle=False) as data:
        n = int(data['num_frames'])
        dt = float(data['meta_control_dt']) if 'meta_control_dt' in data else float(json.loads((directory/'evaluation_protocol.json').read_text())['control_dt'])
        schema = str(data['schema_version'])
        assert schema in ('sonic_episode_v4_multicam', 'twist2_episode_v3_multicam')
        assert n == len(rows) and n > 0
        assert [r['step'] for r in rows] == list(range(n))
        for name in ('frame_index', 'episode_step'):
            if name not in data and schema == 'twist2_episode_v3_multicam':
                continue  # Native TWIST2 uses per-camera indices; no extra RPC metadata.
            assert data[name].shape == (n,), (name, data[name].shape)
            np.testing.assert_array_equal(np.diff(data[name]), np.ones(n-1, dtype=np.int64))
        starts = np.array([r['sim_time'] for r in rows])
        ends = np.array([r['next_sim_time'] for r in rows])
        np.testing.assert_allclose(ends-starts, dt, rtol=0, atol=1e-6)
        np.testing.assert_allclose(starts[1:], ends[:-1], rtol=0, atol=1e-9)
        sonic_shapes = [('encoder_input',(n,1762)), ('encoder_latent',(n,64)),
                            ('decoder_obs',(n,994)), ('final_body_action_29dof',(n,29)),
                            ('vla_action',(n,40)), ('vla_state',(n,64))]
        twist_shapes = [('robot_action_mimic',(n,35)), ('robot_obs_buf',(n,1432)), ('robot_twist2_inference_qpos',(n,29))]
        for name, shape in (sonic_shapes if schema.startswith('sonic') else twist_shapes):
            assert data[name].shape == shape, (name,data[name].shape)
            assert np.isfinite(data[name]).all(), name
        videos = {}
        for camera, prefix in [('ego','vision'), ('g1','vision_world')]:
            np.testing.assert_array_equal(data[prefix+'_frame_indices'], np.arange(n))
            path = Path(str(data[prefix+'_rgb_video_path'].item()))
            if not path.is_absolute():
                path = directory/path
            capture = cv2.VideoCapture(str(path))
            assert capture.isOpened(), path
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
            capture.release()
            assert count == n, (camera,count,n)
            assert abs(fps - 1/dt) < .001, fps
            videos[camera] = dict(path=str(path), decoded_frames=count, fps=fps)
    report = dict(verified=True, native_episode=str(episodes[0]), frames=n,
                  duration_seconds=n*dt, continuous_physics=True, videos=videos)
    (directory/'recording_verification.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
