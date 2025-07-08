import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from multi_agent_package.agents import Agent


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    # super().reset()

    def __init__(self, agents: list, render_mode=None, size=5, perc_num_obstacle = 30 ):

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.perc_num_obstacle = perc_num_obstacle
        self._num_obstacles = int((self.perc_num_obstacle/100)*(self.size*self.size))

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        # self.observation_space = self._make_observation_space() # implement self._make_observation_space()

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = self._make_action_space() # implement self._make_action_space()

        # self._action_to_direction = self._action_to_direction() # implement self._action_to_direction()
        self.agents = agents
        self.num_agents = {'total' : len(self.agents)}

        for ag in self.agents:
                if str(ag.agent_type) in self.num_agents:
                    self.num_agents[str(ag.agent_type)] += 1   
                else:
                    self.num_agents[str(ag.agent_type)] = 1   

        print(self.num_agents) 


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


       
    def _make_observation_space(self):
        observation_space = spaces.Dict({})
        for ag in self.agents:
            observation_space[ag.agent_name] = spaces.Box(0, self.size - 1, shape=(2,), dtype=int)
     
        return observation_space


    # def action_space(self,agent):
    #     agent._make_action_space()
    #     action_space = spaces.Discrete(4)
    #     return action_space
    
    def _make_action_space(self):
        action_space = spaces.Discrete(4)
        return action_space
    

    # def _action_to_direction(self):
    #     """
    #     The following dictionary maps abstract actions from `self.action_space` to
    #     the direction we will walk in if that action is taken.
    #     I.e. 0 corresponds to "right", 1 to "up" etc.
    #     """
        
    #     action_to_direction = {
    #             0: np.array([1, 0]),
    #             1: np.array([0, 1]),
    #             2: np.array([-1, 0]),
    #             3: np.array([0, -1]),}

    #     return action_to_direction
    

    def _get_obs(self):
        obs = {}
        for ag in self.agents:
            obs[ag.agent_name] = ag._get_obs()
        
        return obs

    def _get_info(self):
        info = {}
        for ag in self.agents:
            info[ag.agent_name] = ag._get_info()
        
        return info

    def reset(self, seed=None, options=None,start_location="random"):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self._agents_location = []
        # Choose the agent's location uniformly at random
        for ag in self.agents:
            ag._start_location = self._initialize_start_location(loc=start_location) # implement _initialize_start_obstacle()
            for aj in self.agents:
                aj._start_location = self._initialize_start_location(loc=start_location) # implement _initialize_start_obstacle()
                if aj.agent_name != ag.agent_name:
                    if np.array_equal(ag._start_location, aj._start_location):
                        self.aj = self._initialize_start_location(loc=start_location)

            self._agents_location.append(ag._start_location)
        
        

        self._obstacle_location = self._initialize_obstacle() # implement _initialize_obstacle()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    def _initialize_start_location(self,loc="random"):
        if loc=='random':
            start_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else:
            start_location = np.array(loc)

        return start_location
    
    
    def _initialize_obstacle(self):
        obstacle_location = []
        # for i in range(self._num_obstacles):
        #     temp = self.np_random.integers(0, self.size, size=2, dtype=int)
        #     cond1 = np.array_equal(temp, self._start_location)
        #     cond2 = np.array_equal(temp, self._target_location)
        #     # cond3 = any(np.array_equal(temp, arr) for arr in self._obstacle_location)
        #     while  cond1 or cond2:
        #         temp = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )
        #     obstacle_location.append(temp)
        return obstacle_location
    

    def _get_reward(self):
        # agent_win = np.array_equal(self.agents[agent_id]._agent_location, self.agents[agent_id]._target_location)
        # obstacle_hit = any(np.array_equal(self.agents[agent_id]._agent_location, arr) for arr in self._obstacle_location)

        # # reward logic
        # if obstacle_hit:
        #     reward = -1
        # elif agent_win:
        #     reward = 1
        # else:
        #     reward = 0

        #TBD
        reward = {}
        for ag in self.agents:
            reward[ag.agent_name] = 0

        return reward


    def step(self, action: dict):
       
        agents_mdp = {}
        # agents_mdp = {'obs': None, 'reward' : None, 'done': None, 'trunc' : None, 'info': None}

        for ag in self.agents:
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = ag._actions_to_directions[action[ag.agent_name]]
            
            # We use `np.clip` to make sure we don't leave the grid
            ag._agent_location = np.clip(
                        ag._agent_location + direction, 0, self.size - 1
                        )

        agents_mdp['obs'] = self._get_obs()

        agents_mdp['reward'] = self._get_reward()

        agents_mdp['terminated'] = False

        agents_mdp['trunc'] = False

        agents_mdp['info']  = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return agents_mdp
    

    def render(self):
        if self.render_mode == "rgb_array":
                return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels


        # First we draw the start
        # pygame.draw.rect(
        #     canvas,
        #     (100, 100, 100),
        #     pygame.Rect(
        #         pix_square_size * self._start_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )

            # First we draw the obstacles
        # for i in range(self._num_obstacles):
        #     pygame.draw.rect(
        #         canvas,
        #         (0, 0, 0),
        #         pygame.Rect( # Rect(left, top, width, height) -> Rect
        #             pix_square_size * self._obstacle_location[i], #loc
        #             (pix_square_size, pix_square_size), # size
        #         ),
        #     )

        # Now we draw the agents
        for ag in self.agents:
            ag._draw_agent(canvas,pix_square_size)
           

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def render(self):
        if self.render_mode == "rgb_array":
                return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()