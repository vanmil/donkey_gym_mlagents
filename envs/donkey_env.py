'''
file: donkey_env.py
author: Jelle van Mil
based on: Tawn Kramer
'''

import time
import cv2
import gym
import numpy as np

from mlagents.envs import UnityEnvironment


class ProgressLogger(object):
    """
    Helper function to monitor progress related variables.
    """

    def __init__(self):
        self.episode_nr = -1
        self.total_step = 0

    def reset(self):
        self.episode_nr += 1

    def on_frame(self):
        self.total_step += 1


class DonkeyEnv(gym.Env):
    """
    OpenAI Gym Environment for Donkey
    """

    def __init__(self, level, vae, vae_path, mdnrnn, mdnrnn_path, cam_resolution, simulator_path,
                 simulator_headless, constant_throttle, max_steering,
                 min_throttle, max_throttle, debug_vae, mode, worker_id, reset_params):
        self.level = level
        self.vae = vae
        self.vae_path = vae_path
        self.mdnrnn = mdnrnn
        self.mdnrnn_path = mdnrnn_path
        self.constant_throttle = constant_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.max_steering = max_steering
        self.debug_vae = debug_vae
        self.progress_logger = ProgressLogger()
        self.cam_resolution = cam_resolution  # the resolution that is used by Unity (default 64x64x3)
        self.mode = mode
        self.mdnmode = "hidden"
        self.worker_id = worker_id
        self.cum_reward = 0
        self.last_steering = 0

        if self.constant_throttle:
            print("constant throttle");
            self._action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        else:
            print("no constant throttle");
            self._action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        if self.mode == "rgb":
            self._observation_space = gym.spaces.Box(low=0,
                                                     high=256,
                                                     shape=self.cam_resolution,
                                                     dtype=np.uint8)
        else:
            vector_observation_length = 0
            if self.mode == "vae":
                vector_observation_length += self.vae.latent_dim
            elif self.mode == "vaernn":
                if self.mdnmode == "sample":
                    vector_observation_length += self.vae.latent_dim + self.vae.latent_dim
                elif self.mdnmode == "hidden":
                    vector_observation_length += self.vae.latent_dim + self.mdnrnn.hidden_units
            elif self.mode == "vector":
                vector_observation_length = 1
            # print(vector_observation_length)
            self._observation_space = gym.spaces.Box(low=np.finfo(np.float32).min,
                                                     high=np.finfo(np.float32).max,
                                                     shape=(1, vector_observation_length),
                                                     dtype=np.float32)
        print("observation space:", self._observation_space)
        print("action space:", self._action_space)

        # simulation related variables.
        self.seed()

        if str(simulator_path).lower() == "none":
            simulator_path = None
        self._env = UnityEnvironment(
            simulator_path, worker_id=worker_id, no_graphics=simulator_headless
        )

        self.brain_name = self._env.external_brain_names[0]
        brain = self._env.brains[self.brain_name]
        # high = np.array([1] * brain.vector_action_space_size[0])
        # self._action_space = spaces.Box(-high, high, dtype=np.float32)

        self.reset(config=reset_params)

    def reset(self, config=None):
        """Resets the state of the environment and returns an initial observation.
        In the case of multi-agent environments, this is a list.
        Returns: observation (object/list): the initial observation of the
            space.
        Todo: This return observation does not yet support VAE/RNN conversion.
        """
        time.sleep(0.25)
        info = self._env.reset(config=config)[self.brain_name]
        self.game_over = False
        self.progress_logger.reset()
        if self.mode == "vaernn":
            self.mdnrnn.model.reset_states()

        # obs, reward, done, info = self.step(info)
        if self.mode == "rgb":
            obs = info.visual_observations[0][0]

        else:
            obs = info.vector_observations[0]
        return obs

    def _rescale_throttle(self, throttle):
        # Convert from [-1, 1] to [0, 1]
        throttle_proportion = (throttle + 1) / 2

        # Convert from [0, 1] to [min, max]
        return (1 - throttle_proportion) * self.min_throttle + self.max_throttle * throttle_proportion

    def step(self, action):
        self.progress_logger.on_frame()
        if self.constant_throttle:
            action = np.append(action, self.constant_throttle)
        else:
            action[1] = self._rescale_throttle(action[1])
        action[0] = action[0] * self.max_steering

        ### BEGIN DONT KNOW WHY THIS IS NEEDED
        if self.vae:
            self.vae.load_weights(self.vae_path, by_name=True)
        if self.mode == "vaernn":
            self.mdnrnn.load_weights(self.mdnrnn_path, by_name=False)
        ### END

        info = self._env.step(action)[self.brain_name]
        self._current_state = info

        # observation = ..
        if self.mode == "vae" or self.mode == "vaernn" or self.mode == "rgb":
            observation = info.visual_observations[0][0]
            observation = cv2.resize(observation, self.cam_resolution[:2])
            # observation = observation[44:, :, :]

            if self.vae and (self.mode == "vae" or self.mode == "vaernn"):
                height = observation.shape[0]
                observation = observation[height - self.vae.input_dim[0]:, :, :]
                observation_batchform = np.expand_dims(observation, 0)

                if self.debug_vae and self.progress_logger.episode_nr < 15:
                    encoded = self.vae.encode(observation_batchform)
                    encoded_mean = encoded[0]
                    encoded_logvar = encoded[1]
                    scaled_logvar = (-np.sum(encoded_logvar) - 25) / 50  # 0,32-18/18

                    decoded_mean = self.vae.decode(encoded_mean)[0]
                    rec_error = np.square(np.subtract(observation, decoded_mean)).mean()
                    # decoded_mean[:5, :, :] = 1.0
                    decoded_mean[-6:, :, 0] += rec_error * 100
                    decoded_mean[-13:-7, :, 0] += scaled_logvar
                    # decoded_sample = self.vae.decode(self.vae.encode(observation_batchform)[2])[0]
                    concat = np.concatenate((observation, decoded_mean), axis=1)

                # Encode image into vector
                observation = self.vae.encode(observation_batchform)[0]

                # Add MDNRNN state
                if self.mode == "vaernn" and self.mdnrnn:
                    if self.mdnmode == "sample":
                        rnn_input = np.concatenate((np.expand_dims(observation, 0), action.reshape(1, 1, 2)), axis=2)
                        rnn_mdn_output = self.mdnrnn.predict(rnn_input)  # (mu 32*5 var 32*5 alpha 32 = 352)
                        output_mixture_paramssample = self.mdnrnn.sample_from_output(rnn_mdn_output[0], output_dim=32,
                                                                                     num_mixes=5, temp=0.3,
                                                                                     sigma_temp=0.005)  # 1,32
                        mus, sigs, pi_logits = self.mdnrnn.split_mixture_params(rnn_mdn_output, 32, 5)
                        for i in range(12):
                            rnn_input = np.concatenate(
                                (np.expand_dims(output_mixture_paramssample, 0), action.reshape(1, 1, 2)), axis=2)
                            rnn_mdn_output = self.mdnrnn.predict(rnn_input)
                            output_mixture_paramssample = self.mdnrnn.sample_from_output(rnn_mdn_output[0],
                                                                                         output_dim=32, num_mixes=5,
                                                                                         temp=0.3, sigma_temp=0.005)
                        observation = np.concatenate((observation, output_mixture_paramssample), axis=1)

                        if self.debug_vae and self.progress_logger.episode_nr < 15:
                            decoded_mean_rnn = self.vae.decode(output_mixture_paramssample)[0]
                            concat = np.concatenate((concat, decoded_mean_rnn), axis=1)

                    elif self.mdnmode == "hidden":
                        rnn_input = np.concatenate((np.expand_dims(observation, 0), action.reshape(1, 1, 2)), axis=2)
                        rnn_h_output = self.mdnrnn.predict_hidden(rnn_input)  # 75
                        observation = np.concatenate((observation, rnn_h_output), axis=1)

                        if self.debug_vae:
                            rnn_mdn_output = self.mdnrnn.predict(rnn_input)  # (mu 32*5 var 32*5 alpha 32 = 352)
                            _, sigs, mixes = self.mdnrnn.split_mixture_params(rnn_mdn_output[0], 32, 5)  # 32 *5
                            mixes = self.mdnrnn.softmax(mixes)
                            for i in range(5):
                                sigs[32 * i:32 * i + 1] *= mixes[i]
                            concat[14:20, 64:, 0] += np.sum(sigs) / 50.0 - 0.1

                if self.debug_vae and self.progress_logger.episode_nr < 15:
                    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('image', 600, 300)
                    cv2.imshow('image', concat[:, :, ::-1])

                    cv2.waitKey(10)

        elif self.mode == "vector":
            observation = info.vector_observations[0, :]
        else:
            raise AttributeError("Other modes than 'vae', 'vaernn', 'vector' or 'rgb' not supported")

        reward = info.rewards[0]
        self.cum_reward += reward
        done = info.local_done[0]
        info = {"text_observation": info.text_observations[0], "brain_info": info}

        if reward <= -1.0:
            print("--- Simulation {}: End of episode {} @ step {} --".format(self.worker_id,
                                                                             self.progress_logger.episode_nr,
                                                                             self.progress_logger.total_step))
            if reward < -1.0:
                print(" collision impact: \t", round(reward, 1))
            print(" episode return \t: {}".format(round(self.cum_reward, 1)))

        if done or self.cum_reward >= 400:
            done = True
            self.cum_reward = 0

        self.last_steering = action[0]
        return observation, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self):
        self._env.close()

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self):
        return -float("inf"), float("inf")

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._n_agents
