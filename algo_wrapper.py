# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""algo wrapper"""

import os
import sys
import psutil

from distributed_utils import mpi_fork
from tools import get_default_kwargs_yaml
from evaluator import Evaluator
from ppo_lag import PPOLag
from pdo import PDO
from cpo import CPO
from trpo import TRPO

class AlgoWrapper:
    """Algo Wrapper for algo"""

    def __init__(self, algo, env, parallel=1, custom_cfgs=None):
        self.algo = algo
        self.env = env
        self.env_id = env.env_id
        self.seed = 0  # TOOD
        self.parallel = parallel
        self.custom_cfgs = custom_cfgs
        self.evaluator = None

    def recursive_update(self, args: dict, update_args: dict):
        """recursively update args"""
        for key, value in args.items():
            if key in update_args:
                if isinstance(update_args[key], dict):
                    print(f'{key}:')
                    self.recursive_update(args[key], update_args[key])
                else:
                    # f-strings:
                    # https://pylint.pycqa.org/en/latest/user_guide/messages/convention/consider-using-f-string.html
                    args[key] = update_args[key]
                    menus = (key, update_args[key])
                    print(f'- {menus[0]}: {menus[1]} is update!')
            elif isinstance(value, dict):
                self.recursive_update(value, update_args)

    def learn(self):
        """Agent Learning"""
        # Use number of physical cores as default.
        # If also hardware threading CPUs should be used
        # enable this by the use_number_of_threads=True
        physical_cores = psutil.cpu_count(logical=False)
        use_number_of_threads = bool(self.parallel > physical_cores)
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if mpi_fork(self.parallel, use_number_of_threads=use_number_of_threads):
            # Re-launches the current script with workers linked by MPI
            sys.exit()
        agent = None
        cfgs = get_default_kwargs_yaml(self.algo)
        if self.custom_cfgs is not None:
            self.recursive_update(cfgs, self.custom_cfgs)
        exp_name = os.path.join(self.env.env_id, self.algo)
        cfgs.update(exp_name=exp_name)
        if self.algo == 'PPOLag':
            agent = PPOLag(
                env=self.env,
                exp_name=exp_name,
                data_dir=cfgs['data_dir'],
                seed=cfgs['seed'],
                cfgs=cfgs,)
        if self.algo == 'PDO':
            agent = PDO(
                env=self.env,
                exp_name=exp_name,
                data_dir=cfgs['data_dir'],
                seed=cfgs['seed'],
                cfgs=cfgs,)
        if self.algo == 'CPO':
            agent = CPO(
                env=self.env,
                exp_name=exp_name,
                data_dir=cfgs['data_dir'],
                seed=cfgs['seed'],
                cfgs=cfgs,)
        if self.algo == 'TRPO':
            agent = TRPO(
                env=self.env,
                exp_name=exp_name,
                data_dir=cfgs['data_dir'],
                seed=cfgs['seed'],
                cfgs=cfgs,)
        ac = agent.learn()
        self.evaluator = Evaluator(self.env, ac.pi, ac.obs_oms)

    def evaluate(self, num_episodes: int = 10, horizon: int = 1250, cost_criteria: float = 1.0):
        """Agent Evaluation"""
        assert self.evaluator is not None, 'Please run learn() first!'
        self.evaluator.evaluate(num_episodes, horizon, cost_criteria)

    def render(
            self,
            num_episode: int = 0,
            horizon: int = 1250,
            seed: int = None,
            play=True,
            save_replay_path: str = None,
    ):
        """Render the environment."""
        assert self.evaluator is not None, 'Please run learn() first!'
        self.evaluator.render(num_episode, horizon, seed, play, save_replay_path)
