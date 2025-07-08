import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class Agent(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    # super().reset()

    def __init__(self, agent_type, agent_name):

        # self.size = size  # The size of the square grid
        # self.window_size = 512  # The size of the PyGame window
        # self.perc_num_obstacle = perc_num_obstacle
        # self._num_obstacles = int((self.perc_num_obstacle/100)*(self.size*self.size))

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.agent_type = agent_type
        self.agent_name = agent_name

        # self.observation_space = self._make_observation_space() # implement self._make_observation_space()

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = self._make_action_space() # implement self._make_action_space()

        self._actions_to_directions = self._action_to_direction() # implement self._action_to_direction()

        self._agent_location  = np.array([0, 0])

       
    # def _make_observation_space(self):
    #     observation_space = spaces.Dict(
    #             {
    #                 "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int)
    #             }
    #         )
    #     return observation_space

    def _make_action_space(self):
        action_space = spaces.Discrete(4)
        return action_space
    

    def _action_to_direction(self):
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        action_to_direction = {
                0: np.array([1, 0]),
                1: np.array([0, 1]),
                2: np.array([-1, 0]),
                3: np.array([0, -1]),}

        return action_to_direction
    

    def _get_obs(self,global_obs=None):
        return { 'local' : self._agent_location,
                 'global' : global_obs}

    def _get_info(self):
        return {
            "<placeholder>": self.agent_name
        }


    # def _initialize_start_location(self,loc="random"):
    #     if loc=='random':
    #         start_location = self.np_random.integers(0, self.size, size=2, dtype=int)
    #     else:
    #         start_location = np.array(loc)

    #     self._agent_location

    #     return start_location
    

    # def _reward_system(self,agent_location): ---> implement it in gridworld
    #     agent_win = np.array_equal(self._agent_location, self._target_location)
    #     obstacle_hit = any(np.array_equal(self._agent_location, arr) for arr in self._obstacle_location)

    #     # reward logic
    #     if obstacle_hit:
    #         reward = -1
    #     elif agent_win:
    #         reward = 1
    #     else:
    #         reward = 0

    #     return reward


    # def step(self, action):
    #     # Map the action (element of {0,1,2,3}) to the direction we walk in
    #     direction = self._action_to_direction[action]
    #     # We use `np.clip` to make sure we don't leave the grid
    #     self._agent_location = np.clip(
    #         self._agent_location + direction, 0, self.size - 1
    #     )

    #     # An episode is done iff the agent has reached the target
    #     terminated = np.array_equal(self._agent_location, self._target_location)

    #     reward = self._reward_system(self._agent_location) # implement _reward_system()

    #     observation = self._get_obs()

    #     info = self._get_info()

    #     if self.render_mode == "human":
    #         self._render_frame()

    #     return observation, reward, terminated, False, info
    

    def _draw_agent(self,canvas,pix_square_size):

        if self.agent_type=="predator":
            color = (255,0,0)
        elif self.agent_type=="prey":
            color = (0,255,0)
        else:
            color = (0,0,255)

       
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            color,
            (self._agent_location + 0.5) * pix_square_size, #fix this
            pix_square_size / 3,
        )



    # def close(self):
    #     if self.window is not None:
    #         pygame.display.quit()
    #         pygame.quit()