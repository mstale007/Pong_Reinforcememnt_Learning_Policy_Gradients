""" Dependencies"""
import numpy as np
import pickle as pickle
import gym
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib import animation
from JSAnimation.IPython_display import display_animation

# Hyperparameters: Here the hyperparameters are decided by reffering to some literature
H = 200 # Number of neurons in the hidden layer
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4 #Learning Rate of our neural network

"""The actual model does something more effective than updating weights negatively or positively according to rewards,
it uses discounted reward the detailed reasoning is given here:
https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning"""
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True# resume from previous checkpoint?
render = False

"""The size of game screen provided by Gym is 210x160x3 from which we are cropping 80x80 part,
and passsing it our neural network after Flattening"""
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
    #Loading the model weights and building on top of it..
    #This becomes very important when you want to train your model for days
  model = pickle.load(open('save.p', 'rb'))
else:
    #Starting from sctratch making a dictionary having arrays as
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } #Stores gradient that are to be updated after every batch update
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  #print(discounted_r)
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """Backpropogation. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # Applying relu as activation function
  dW1 = np.dot(dh.T, epx)
  a=model['W1'][:80,:80]
  return {'W1':dW1, 'W2':dW2}

def display_frames_as_gif(frames, filename_gif = None):
    """
    Making GIF from continuous photos
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename_gif:
        anim.save(filename_gif, writer = 'imagemagick', fps=50)
    display(display_animation(anim, default_mode='loop'))


#Initialize environment for Pong
env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used for computing the difference frame
xs,hs,error,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
frames=[]
while True:
  env.render()
  ##frame = env.render(mode = 'rgb_array')
  #frames.append(frame)
  #fig,ax = plt.subplots()
  #firstframe = env.render(mode = 'rgb_array')
  #im = ax.imshow(firstframe)
  #frame = env.render(mode = 'rgb_array')
  #im.set_data(frame)

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if 0.5 < aprob else 3
  """Here instead of 0.5 you should use a random number
  from gaussian distribution this lets the model explore more and be more accurate at its decesion
   action = 2 if np.random.uniform() < aprob else 3"""
  # record various observations (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  error.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  #Taking next step in the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished (i.e. either one of them has won)
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    eperror = np.vstack(error)
    epr = np.vstack(drs)
    xs,hs,error,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    #print(discounted_epr)

    eperror *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    print(eperror)
    grad = policy_backward(eph, eperror)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        #display_frames_as_gif(frames, filename_gif="manualplay.gif")
        #frames=[]

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    #print('resetting env. episode reward total was %f. running mean: %f',%(reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f'%(episode_number, reward),end=' ')
    print(str('' if reward == -1 else ' !!!!!!!!'))
