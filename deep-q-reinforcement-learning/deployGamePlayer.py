import tensorflow as tf
from tensorflow import keras

# common imports
import numpy as np

# creating the wrapped Atari Environment
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

# load the non frame-skipped variant of Enduro (with these parameters)
max_episode_steps = 27000 # load 108k frames from the ALE (1 step = 4 frames)
environment_name = "EnduroNoFrameskip-v0"

# load the environment
env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4]
)

# wrap environment in a TFPyEnvironment for usability within a TF graph
from tf_agents.environments.tf_py_environment import TFPyEnvironment
tf_env = TFPyEnvironment(env)

from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer


class ShowProgress:
  def __init__(self, total):
    self.counter = 0
    self.total = total
  def __call__(self, trajectory):
    if not trajectory.is_boundary():
      self.counter += 1
    if self.counter % 100 == 0:
      print("\r{}/{}".format(self.counter, self.total), end="")

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

saved_policy = tf.compat.v2.saved_model.load('savedPolicy')

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

watch_driver = DynamicStepDriver(
    tf_env,
    saved_policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=3000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)