

import ray
import gym

from gym.wrappers import AtariPreprocessing, FrameStack


from rsn.replay_memory import ReplayMemory
from rsn.parameter_server import ParameterServer
from rsn.actor import Actor

from rsn.hyper_parameter import REPLAY_MEMORY_SIZE

@ray.remote(num_cpus=1)
def test_actor(eps, replay_memory, parameter_server):
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env)
    env = FrameStack(env, num_stack=4)

    actor = Actor(env, env.action_space.n, eps, replay_memory, parameter_server)

    while True:
        actor.run_episode()


if __name__ == "__main__":
    ray.init()
    
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    parameter_server = ParameterServer()

    ref = test_actor.remote(0.5, replay_memory, parameter_server)

    import time

    while True:
        time.sleep(5.0)
        print(replay_memory.size())