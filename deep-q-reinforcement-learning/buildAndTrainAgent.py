# build an train a RL DQN to play the game enduro

import tensorflow as tf
from tensorflow import keras

# common imports
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)

# load enduro from the Atari Suite Gym
from tf_agents.environments import suite_gym
env = suite_gym.load("Enduro-v0")

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

# creating the Deep Q Network
from tf_agents.networks.q_network import QNetwork
preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params = [512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params
)

#creating the DQN agent
from tf_agents.agents.dqn.dqn_agent import DqnAgent

train_step = tf.Variable(0) # training step counter
update_period = 4 # train model every 4 steps

optimizer = keras.optimizers.RMSprop(lr=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True) # optimizer created with 2015 DQN Paper's Hyperparameters

epsilon_fn = keras.optimizers.schedules.PolynomialDecay( # This function computes the epsilon value according to the greedy policy described earlier
    initial_learning_rate=1.0,
    decay_steps=250000 // update_period,
    end_learning_rate=0.01
)

agent = DqnAgent(tf_env.time_step_spec(), # DQN agent takes the time steps and action and optimizer and returns the epsilon to help make next choice
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000,
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99,
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step)
                 )

agent.initialize() 
# creating the Replay Buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=50000
)

# observer for buffer --  could be manually implemented
replay_buffer_observer = replay_buffer.add_batch

class ShowProgress:
  def __init__(self, total):
    self.counter = 0
    self.total = total
  def __call__(self, trajectory):
    if not trajectory.is_boundary():
      self.counter += 1
    if self.counter % 100 == 0:
      print("\r{}/{}".format(self.counter, self.total), end="")

# creating Training Metrics
from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

# create driver to broadcast experiences to observer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps = update_period
)

# creating a random policy to initialize from so the driver can explore the area
from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000
)

final_time_Step, final_policy_state = init_driver.run()

# sampling the dataset trajectories from the replay buffer
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

# creating the training loop
from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

checkpoint = tf.train.Checkpoint(agent = agent)

# this function runus the training loop for n_iterations
def train_agent(n_iterations):
  time_step = None
  policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
  iterator = iter(dataset)
  for iteration in range(n_iterations):
    time_step, policy_state = collect_driver.run(time_step, policy_state)
    trajectories, buffer_info = next(iterator)
    train_loss = agent.train(trajectories)
    print("\r{} loss:{:.5f}".format(
        iteration, train_loss.loss.numpy()), end="")
    if iteration % 1000 == 0:
      log_metrics(train_metrics)

train_agent(2000)

checkpoint.save("lastModelCheckpoint")

from tf_agents.policies.policy_saver import PolicySaver
PolicySaver(agent.policy).save("savedPolicy")

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

prev_lives = tf_env.pyenv.envs[0].ale.lives()
def reset_and_fire_on_life_lost(trajectory):
    global prev_lives
    lives = tf_env.pyenv.envs[0].ale.lives()
    if prev_lives != lives:
        tf_env.reset()
        tf_env.pyenv.envs[0].step(np.array(1))
        prev_lives = lives

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

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=3000)
final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)

import PIL
import os

image_path = os.path.join("myAgentPlays.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)