from gym.envs.registration import register
from donkey_gym_mlagents.version import VERSION as __version__

register(
    id='donkey-simulator-v1',
    entry_point='donkey_gym_mlagents.envs.donkey_env:DonkeyEnv',
    kwargs={'level': 0,
            'vae': None,
            'vae_path': None,
            'mdnrnn': None,
            'mdnrnn_path': None,
            'cam_resolution': (64, 64, 3),
            'simulator_path': 'donkey_gym_mlagents/envs/simulation_binary/donkey_sim.x86_64',
            'simulator_headless': False,
            'constant_throttle': None,
            'min_throttle': 0.0,
            'max_throttle': 0.75,
            'max_steering': 0.75,
            'debug_vae': False,
            'mode': 'vae',
            'worker_id': 0,
            'reset_params': {},
            }
)