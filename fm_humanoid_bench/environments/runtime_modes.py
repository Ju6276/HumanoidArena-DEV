"""Use the original HA visual randomization hooks in every runtime route."""

def setup_vision_test_light_from_cfg(env_cfg):
    config = getattr(env_cfg, 'vision_randomization', None)
    if isinstance(config, dict) and config.get('enabled', False):
        from tasks.common_runtime import setup_vision_test_light
        setup_vision_test_light(config.get('prim_path','/World/light'))


def apply_vision_light_randomization_from_cfg(env_cfg, *, episode_seed=None, seed_source=None):
    config = getattr(env_cfg, 'vision_randomization', None)
    if not isinstance(config,dict) or not config.get('enabled',False):
        return {'enabled':False}
    from tasks.common_runtime import apply_vision_light_randomization
    seed = int(getattr(env_cfg,'seed',0) if episode_seed is None else episode_seed)
    params = {key:config.get(key) for key in ('intensity_range','color_range','position_range')}
    params.update(prim_path=config.get('prim_path','/World/light'),episode_seed=seed,
                  rotation_ranges=config.get('rotation'))
    apply_vision_light_randomization(**params)
    return {'enabled':True,'seed':seed,'parameters':params}
