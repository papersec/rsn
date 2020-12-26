import ray
import gym

from gym.wrappers import AtariPreprocessing, FrameStack


from rsn.replay_memory import ReplayMemory
from rsn.parameter_server import ParameterServer
from rsn.actor import Actor
from rsn.learner import Learner

from rsn.hyper_parameter import REPLAY_MEMORY_SIZE

@ray.remote(num_cpus=0.4)
def test_actor(i, eps, replay_memory, parameter_server):
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariPreprocessing(env)
    env = FrameStack(env, num_stack=4)

    assert env.action_space.n == 4

    actor = Actor(env, env.action_space.n, eps, replay_memory, parameter_server)

    t = 0
    score_sum = 0.0
    while True:
        t += 1
        score_sum += actor.run_episode()
        if t % 100 == 0:
            print("Actor", i, "got", score_sum / 100)

@ray.remote
def test_learner(n_action, replay_memory, parameter_server):
    learner = Learner(n_action, replay_memory, parameter_server)

    import time
    while True:
        time.sleep(5.0)
        s = replay_memory.size()

        if s >= 1000:
            break

    learner.run()


if __name__ == "__main__":
    ray.init()
    
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    parameter_server = ParameterServer()

    actors = []

    num_actor = 6
    for i in range(num_actor):
        eps = 0.4 ** (1 + i / (num_actor - 1) * 7)
        actor = test_actor.remote(i, eps, replay_memory, parameter_server)
        actors.append(actor)

    ref_ = test_learner.remote(4, replay_memory, parameter_server)

    import time

    while True:
        time.sleep(10.0)
        s = replay_memory.size()
        print("Replay Memory Size", s)