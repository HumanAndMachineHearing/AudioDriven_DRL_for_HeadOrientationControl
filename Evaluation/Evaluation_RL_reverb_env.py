# ------------------------------------------------------------
# Import packages and libraries
# ------------------------------------------------------------

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
import os
import wandb

#%%
# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
dir_random_sampling_file = "/fldr_randomsampling"
dir_data = "/fldr_data"
dir_brir = "/fldr_brir"  # TESTING ENVIRONMENT (has to be reverberant here)
dir_model = "/fldr_weights"  # TRAINING ENVIRONMENT
name_model = "name_weights.pth"  # TRAINING ENVIRONMENT
name_sampling_file = "samplingfile_name.txt"  # TESTING ENVIRONMENT (has to be reverberant here)

# #%%
# ------------------------------------------------------------
# Specifications and preliminaries
# ------------------------------------------------------------
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

# Parameters for reinforcement learning 
gamma = 0.8  # discount factor
learning_rate = 0.001
batch_size = 1024
reward_optimal_choice = 0.1
reward_suboptimal_choice = 0
reward_bad_choice = -0.2
reward_target_reached = 1
memory_capacity = 2000  # number of experiences to store per location
rms_target = 10  # RMS scaling target for data normalisation
clipgradients = 0  # 0 = no, 1 = yes

# Locations: for the BRIR, 270 = left and 90 is right --> this may need to change to a two-dimensional matrix in case we decide to restrict the field of view in a different way
az_angles = ["090", "075", "060", "045", "030", "015", "000", "345", "330", "315", "300", "285",
             "270"]  # head orientation
DOA_az_angles = ["-90", "-75", "-60", "-45", "-30", "-15", "000", "015", "030", "045", "060", "075", "090"]
el_angles = ["-45", "-20", "000", "020", "045"]
DOA_el_angles = ["-40", "-20", "000", "020",
                 "040"]  # elevation deviations are in equal steps to avoid further complications, this means that dev -25 and -20 are added to the same point in the buffer

all_locs_for_buffer = list()  # initialise empty list
for i in range(len(DOA_el_angles)):
    for j in range(len(DOA_az_angles)):
        all_locs_for_buffer.append([DOA_el_angles[i], DOA_az_angles[j]])


#%%
# ------------------------------------------------------------
# Functions miscellaneous
# ------------------------------------------------------------
def initial_states_to_list(rand_samp_file):
    # this function generates list "initial states" which contains for each line in the random sampling file brir and
    # the file name

    initial_states = list()
    f = open(rand_samp_file, "r")
    for line in f:
        brir_temp, file = line.split(" ")  # this returns the brir name and the speaker sample
        initial_states.append((brir_temp, file.removesuffix("\n")))
    f.close()

    return initial_states


def chebyshev_distance(hor_az, hor_el, target_az, target_el):
    # Function to calculate Chebyshev distance between current orientation and target orientation (az = 0, el = 0)

    # retrieve azimuth and elevation deviation
    cur_az_deviation = calc_doa_az(int(target_az), int(hor_az))
    cur_el_deviation = calc_doa_el(int(target_el), int(hor_el))
    # convert to string and format name to three characters
    if len(str(cur_az_deviation)) == 1:
        cur_az_deviation = '00' + str(cur_az_deviation)
    elif len(str(cur_az_deviation)) == 2:
        cur_az_deviation = '0' + str(cur_az_deviation)
    if len(str(cur_el_deviation)) == 1:
        cur_el_deviation = '00' + str(cur_el_deviation)
    elif len(str(cur_el_deviation)) == 2:
        cur_el_deviation = '0' + str(cur_el_deviation)

    dist_az = DOA_az_angles.index(str(cur_az_deviation))
    dist_el = DOA_el_angles.index(str(cur_el_deviation))

    # calculate Chebyshev distance
    dist = np.max([np.abs(dist_az - 6), np.abs(dist_el - 2)])

    return int(dist)


def rms_norm(arr, rms_target):
    # Normalises monaural sound clip to a given RMS

    if type(arr) == torch.Tensor:
        #arr = arr.numpy()  # convert to numpy array
        rms_temp = torch.sqrt(torch.sum(torch.square(arr)) / len(arr))  # calculate current RMS
        rms_scale_fact = rms_target / rms_temp  # calculate scaling factor
        arr = arr * rms_scale_fact  # multiply original sample with scaling factor
    elif type(arr) == np.ndarray:
        rms_temp = np.sqrt(np.sum(np.square(arr)) / len(arr))  # calculate current RMS
        rms_scale_fact = rms_target / rms_temp  # calculate scaling factor
        arr = arr * rms_scale_fact  # multiply original sample with scaling factor

    return arr


def get_brir_headrotation(brir_name):
    return brir_name.split("_")[16], brir_name.split("_")[18][:3]


def calc_doa_az(target_hor, current_hor):
    # specify for flipping around, if heador is to the right of target, positive difference; if heador is to the left
    # of target, negative difference
    if target_hor >= 270 and current_hor <= 90:
        current_hor += 360
    elif target_hor <= 90 and current_hor >= 270:
        current_hor -= 360

    doa_az = target_hor - current_hor

    return doa_az


def calc_doa_el(target_hor, current_hor):
    # if heador is above the target, difference is negative; if heador is below the target, difference is positive
    # to avoid further complications, change into equidistant steps
    if current_hor == 45:
        current_hor = 40
    elif current_hor == -45:
        current_hor = -40

    if target_hor == 45:
        target_hor = 40
    elif target_hor == -45:
        target_hor = -40

    doa_el = target_hor - current_hor

    return doa_el


#%%
# ------------------------------------------------------------
# Deep Q network
# ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_observations, n_angles):
        super(DQN, self).__init__()
        self.gru1 = nn.GRU(n_observations, 512, 1, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(512, 256, 1, batch_first=True, bidirectional=False)
        self.gru3 = nn.GRU(256, 128, 1, batch_first=True, bidirectional=False)
        self.gru4 = nn.GRU(128, 64, 1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(64 * 2, n_angles)
        self.dropout20 = nn.Dropout(p=0.2)
        self.dropout50 = nn.Dropout(p=0.5)

    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout20(x)
        x, _ = self.gru2(x)
        x = self.dropout20(x)
        x, _ = self.gru3(x)
        x = self.dropout20(x)
        x, _ = self.gru4(x)
        x = self.dropout50(x)
        x = torch.cat((x[:, 0, :], x[:, 1, :]), dim=1)
        x = self.fc(x)
        return x


#%%
# ------------------------------------------------------------
# Environment functions
# ------------------------------------------------------------
def open_brir(brir_name):
    """
    Load BRIR 
    """
    brir, brir_fs = torchaudio.load(os.path.join(dir_brir, brir_name), format="wav")
    brir = brir[:, ::3]  # downsample to 16 kHz
    return brir


class environment():

    def __init__(self, fs, gamma, lr, batsize, n_windows, length_windows, az_angles, el_angles, random_sampling_file,
                 policy_network, n_actions, actions, device, rew_opt_ch, rew_subopt_ch,
                 rew_bad_ch, clipgrads):
        self.device = device
        self.fs = fs
        self.n_windows = n_windows
        self.length_windows = length_windows
        self.az_angles = az_angles
        self.el_angles = el_angles
        self.DOA_az_angles = DOA_az_angles
        self.DOA_el_angles = DOA_el_angles
        self.init_states = initial_states_to_list(random_sampling_file)
        self.policy_network = policy_network.to(device)
        self.action_space = actions
        self.n_actions = n_actions
        self.batch_size = batsize
        self.criterion = nn.SmoothL1Loss()
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=self.lr, amsgrad=True)
        self.gamma = gamma
        self.reward_optimal_choice = rew_opt_ch
        self.reward_suboptimal_choice = rew_subopt_ch
        self.reward_bad_choice = rew_bad_ch
        self.clipgrads = clipgrads

    def get_initial_state(self, idx):
        """
        Gives a new sample and initial BRIR/state for a new epoch.
        idx: the index of the new epoch.
        Returns: a tuple with the new BRIR and new sample.
        """
        return self.init_states[idx]

    def open_sample_split(self, sample_name):
        """
        Gets the data of the sample, and splits it into the amount of specified windows.
        sample_name: the name of the sample.
        Returns: a 2D array with the sample split into windows.
        """
        sample, _ = torchaudio.load(os.path.join(dir_data, sample_name), format="flac")
        sample = sample  #[:,::2] # downsample sound to 8 kHz
        return sample.reshape((self.n_windows, -1))

    def convolve_sound(self, window, hrtf):
        """
        Convolves the BRIR with the window, and cuts result at the size of the window length.
        window: the window of the sample that should be convolved.

        """
        conv_sound = torchaudio.functional.convolve(window.repeat([2, 1]), hrtf)
        conv_sound = conv_sound[:, ::2]  # downsample from 16 kHz to 8 khz
        conv_sound = conv_sound[:, :self.length_windows]  # cut to right duration after convolution

        return conv_sound

    def get_Q_values_from_state(self, observation):
        """
        Puts observation into target network and chooses actions
        observation: the sample convolved with the HRTF
        """
        with torch.no_grad():
            qvals = self.policy_network(observation)

        return qvals



    def get_best_actions_from_Q_values(self, Q_values):
        """
        Chooses the best action index given the Q-values for the azimuth and elevation angle rotation.
        Q_values: the Q-values as returned by the target network.
        Returns: index of the best action
        """
        return torch.argmax(Q_values)
        # return torch.min(torch.floor(torch.add(Q_values, 1)*(self.n_directions/2)),torch.tensor(self.n_directions-1))

    def sample_action_epsilon_greedily(self, best_action, epsilon):
        """
        Samples an action epsilon-greedily.
        best_actions: the index of the best actions.
        epsilon: the greediness parameter.q
        Returns: the actions [az,el].
        """
        p_best_action = 1 - epsilon + epsilon / self.n_actions
        p_action = epsilon / self.n_actions
        probability_table = np.full(self.n_actions, p_action)  # assigns the probability of each action
        probability_table[best_action] = p_best_action  # assigns higher probability to best action
        index = np.random.choice(np.arange(self.n_actions),
                                 p=probability_table)  # randomly selects an action based on the probabilities in the table
        return self.action_space[index], index

    def take_action(self, current_action, hor_az, hor_el, tar_az, tar_el):
        """
        Finds the next head orientation (azimuth and elevation) after taking the action. 
        actions: a tensor with two values for the azimuth and elevation angles.
        Returns: the new head orientation (azimuth, elevation), after taking the action, as well as the new DOA.
        """

        # retrieve DOAs and convert to string
        cur_az_deviation = str(calc_doa_az(int(tar_az), int(hor_az)))
        cur_el_deviation = str(calc_doa_el(int(tar_el), int(hor_el)))
        # format name to three characters
        if len(cur_az_deviation) == 1:
            cur_az_deviation = '00' + cur_az_deviation
        elif len(cur_az_deviation) == 2:
            cur_az_deviation = '0' + cur_az_deviation
        if len(cur_el_deviation) == 1:
            cur_el_deviation = '00' + cur_el_deviation
        elif len(cur_el_deviation) == 2:
            cur_el_deviation = '0' + cur_el_deviation

        current_DOA_az_angle_idx = self.DOA_az_angles.index(cur_az_deviation)  # find the index of the current DOA
        current_DOA_el_angle_idx = self.DOA_el_angles.index(cur_el_deviation)
        current_az_angle_idx = self.az_angles.index(hor_az)  # retrieve current head orientation
        current_el_angle_idx = self.el_angles.index(hor_el)

        az_mov, el_mov = current_action  # split selected action in azimuth and elevation

        # this specifies that the agent cannot go out of bounds, this may need to be adjusted for brirs because going out of bounds varies per trial in case we decide to do this
        # if the agent wants to go out of bounds, it stays in the same head orientation 
        # azimuth
        if current_DOA_az_angle_idx + az_mov >= len(self.DOA_az_angles):
            az_mov = 0
        if current_DOA_az_angle_idx + az_mov < 0:
            az_mov = 0
        if current_az_angle_idx + az_mov >= len(self.az_angles):
            az_mov = 0
        if current_az_angle_idx + az_mov < 0:
            az_mov = 0

        # elevation; head orientation goes in opposite direction of DOA movement, therefore - el_mov for head orientation and +el_mov for DOA    
        if current_DOA_el_angle_idx + el_mov >= len(
                self.DOA_el_angles):  # first check whether elevation goes out of bounds in terms of DOA
            el_mov = 0
        if current_DOA_el_angle_idx + el_mov < 0:
            el_mov = 0
        if current_el_angle_idx - el_mov >= len(
                self.el_angles):  # now check whether head orientation goes out of bounds
            el_mov = 0
        if current_el_angle_idx - el_mov < 0:
            el_mov = 0

        # this is for the checking to make sure that there is no problem between -45 and -40
        if int(hor_el) == 45:
            hor_el = 40
        elif int(hor_el) == -45:
            hor_el = -40

        # Calculate new head orientations and DOAs
        # head orientation goes in the opposite direction of DOA movement, therefore - elmov for current_el_angle_idx but + el move for current DOA_el_angle_idx    
        new_hor_el = self.el_angles[current_el_angle_idx - el_mov]
        new_hor_az = self.az_angles[current_az_angle_idx + az_mov]
        new_DOA_az = current_DOA_az_angle_idx + az_mov  #this is an index
        new_DOA_el = current_DOA_el_angle_idx + el_mov  # this is an index

        return new_hor_az, new_hor_el, new_DOA_az, new_DOA_el

    def get_reward(self, hor_az_selectedaction, hor_el_selectedaction, old_hor_az, old_hor_el, tar_az, tar_el):
        #get_reward(self, new_hor_az, new_hor_el, old_hor_az, old_hor_el, DOA_az_start, DOA_el_start):
        """
        Gives reward based on the orientation deviation
        """
        # calculate DOA of selected action 
        cur_az_deviation = str(calc_doa_az(int(tar_az), int(hor_az_selectedaction)))
        cur_el_deviation = str(calc_doa_el(int(tar_el), int(hor_el_selectedaction)))
        # make sure name consists of three characters 
        if len(cur_az_deviation) == 1:
            cur_az_deviation = '00' + cur_az_deviation
        elif len(cur_az_deviation) == 2:
            cur_az_deviation = '0' + cur_az_deviation
        if len(cur_el_deviation) == 1:
            cur_el_deviation = '00' + cur_el_deviation
        elif len(cur_el_deviation) == 2:
            cur_el_deviation = '0' + cur_el_deviation
            # calculate DOA of old head orientation
        old_az_deviation = str(calc_doa_az(int(tar_az), int(old_hor_az)))
        old_el_deviation = str(calc_doa_el(int(tar_el), int(old_hor_el)))
        # make sure name consists of three characters 
        if len(old_az_deviation) == 1:
            old_az_deviation = '00' + old_az_deviation
        elif len(old_az_deviation) == 2:
            old_az_deviation = '0' + old_az_deviation
        if len(old_el_deviation) == 1:
            old_el_deviation = '00' + old_el_deviation
        elif len(old_el_deviation) == 2:
            old_el_deviation = '0' + old_el_deviation

            # first check whether target is reached and if so, assign reward = 1
        if cur_az_deviation == "000" and cur_el_deviation == "000":
            return 1
        else:  # if target is not reached, calculate distances for all possible actions to find whether it is optimal or suboptimal action
            dists = np.zeros(n_actions)
            for i, action in enumerate(
                    self.action_space):  # cycle through all possible actions to compute distance for each action
                temp_hor_az, temp_hor_el, temp_doa_az, temp_doa_el = self.take_action(action, old_hor_az, old_hor_el,
                                                                                      int(tar_az),
                                                                                      int(tar_el))  # retrieve new heador (az, el) for this particular action
                #dists[i] = np.sqrt(int(self.DOA_az_angles[temp_doa_az])**2 + int(self.DOA_el_angles[temp_doa_el])**2)
                dists[i] = np.sqrt((int(temp_doa_az) - int(self.DOA_az_angles.index("000"))) ** 2 + (
                            int(temp_doa_el) - int(self.DOA_el_angles.index("000"))) ** 2)
            optimal_dist = np.min(dists)  # retrieve smallest distance across all possible actions
            # retrieve actual distance for this particular action 
            # actual_dist = np.sqrt((self.az_angles.index(new_hor_az) - self.az_angles.index(DOA_az_start))**2 + (self.el_angles.index(new_hor_el) - self.el_angles.index(DOA_el_start))**2)
            actual_dist = np.sqrt(
                (self.DOA_az_angles.index(cur_az_deviation) - self.DOA_az_angles.index("000")) ** 2 + (
                            self.DOA_el_angles.index(cur_el_deviation) - self.DOA_el_angles.index("000")) ** 2)
            # retrieve old distance for the comparison
            old_dist = np.sqrt((self.DOA_az_angles.index(old_az_deviation) - self.DOA_az_angles.index("000")) ** 2 + (
                        self.DOA_el_angles.index(old_el_deviation) - self.DOA_el_angles.index("000")) ** 2)

            if optimal_dist == actual_dist:
                return self.reward_optimal_choice  # Optimal distance improvement
            elif actual_dist < old_dist:
                return self.reward_suboptimal_choice  # Suboptimal distance improvement
            else:
                return self.reward_bad_choice  # Distance gets larger

    def next_state(self, current_brir, new_heador_az, new_heador_el):
        """
        Finds the name of the next BRIR.
        current_brir: the name of the current brir
        new_az: the new azimuth angle after the action.
        el_mov: the new elevation angle after the action.
        Returns: the filename of the BRIR.
        """

        new_brir_name = current_brir.split("_")
        new_brir_name[16] = new_heador_az
        new_brir_name[18] = new_heador_el + ".wav"

        return "_".join(new_brir_name)

    def is_terminal(self, new_hor_az, new_hor_el, tar_az, tar_el):
        """
        Returns a boolean testing whether the agent has reached the target (and thus is terminal)
        """
        return new_hor_az == tar_az and new_hor_el == tar_el


#%%
# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def norm_length_episode(ep_length, start_az, start_el):
    # this function calculates the length per episode normalised for the starting distance, as episodes can be closer or further away from the target

    # calculate starting distance (i.e. shortest path)
    start_dist = chebyshev_distance(0, 0, start_az, start_el)  # the starting head orientation is always 0,0
    # normalized length per episode = episode length / starting distance where a length of 1 = shortest possible path
    norm_length = np.round(ep_length / start_dist,
                           2)  # here ep_length is the final episode number (i.e. after terminal state)

    return norm_length


def norm_reward_episode(ep_reward, start_az, start_el, reward_opt_ch, reward_tar_reach):
    # calculate starting distance (i.e. shortest path)
    start_dist = chebyshev_distance(0, 0, start_az, start_el)  # the starting head orientation is always 0,0
    # calculate maximum reward as shortest path * reward + final reward for reaching targte
    max_r = (start_dist * reward_opt_ch) + reward_tar_reach
    # calculate normalised reward as the actual reward divided by the maximum reward where norm_r = 1 means maximum reward achieved
    norm_r = np.round(ep_reward / max_r, 2)

    return norm_r


#%%
# ------------------------------------------------------------
# Parameters RL network
# ------------------------------------------------------------
n_observations = 4000
n_actions = 8
actions = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

# use the same seed
torch.manual_seed(42)  # instead of 42
policy_network = DQN(n_observations, n_actions).to(device)
policy_network.load_state_dict(torch.load(os.path.join(dir_model, name_model),map_location=torch.device(device)))
#%%
# ------------------------------------------------------------
# Make environment
# ------------------------------------------------------------

fs = 8000  # set sampling rate
n_windows = 20  # set the number of windows
length_windows = 4000

random_sampling_file = os.path.join(dir_random_sampling_file, name_sampling_file)

env = environment(fs, gamma, learning_rate, batch_size, n_windows, length_windows, az_angles, el_angles,
                  random_sampling_file, policy_network, n_actions, actions, device,
                  reward_optimal_choice, reward_suboptimal_choice, reward_bad_choice, clipgradients)

#%%
# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
# Specifications
episodes = len(env.init_states)  #set maximum number of episodes, shouldn't this be done with early stopping? how does that work for RL?
epsilon_zero = 0
epsilon = epsilon_zero  # original assignment

# Initialize variables to track metrics 
episode_length = list()  #
episode_length_norm = list()
end_distances = list()  #
trajectories = list()  #
all_rewards_raw = list()
all_rewards_norm = list()  # list to store normalized rewards
all_epsilon = list()
DOA_in_room_heador_az0el0 = list()  # DOA of sound source at start of episode

# Set parameters for WandB logging 
wandb.init(project="RL_final_runs_test")  # set the wandb project where this run will be logged
projectname = 'Snellius_BRIR_wRMSnorm_lowreverb_MemCap' + str(memory_capacity) + '_Eps' + str(
    epsilon_zero) + '_BatSi_' + str(batch_size) + '_Disc' + str(
    gamma) + '_RewOpt' + str(reward_optimal_choice) + '_RewSOpt' + str(
    reward_suboptimal_choice) + '_RewBad' + str(reward_bad_choice) + '_LR' + str(learning_rate) + '_Clip' + str(
    clipgradients)
wandb.run.name = projectname

wandb.config = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "architecture": "RNN",
    "dataset": "Wessel_RNN_RL_data",
    "epochs": episodes,
}

policy_network.eval()
# Loop through every episode 
for ep in tqdm(range(episodes)):

    # Initialize variables for this episode
    trajectory = list()  # initialize trajectory
    total_reward = 0  # intialise total reward calculation for each episode

    # Initialize environment
    brir_name, samp_name = env.get_initial_state(ep)  # retrieve name brir and speaker sample
    DOA_in_room_heador_az0el0.append((brir_name.split("_")[12], brir_name.split("_")[
        14]))  # this retrieves the Az position and the El position of the sound with respect to the listener (original DOA)
    heador_az, heador_el = get_brir_headrotation(brir_name)  # extract head orientation

    tar_azimuth = DOA_in_room_heador_az0el0[ep][0]  # store starting values of DOA azimuth and elevation
    tar_elevation = DOA_in_room_heador_az0el0[ep][1]  # store starting values of DOA azimuth and elevation

    # Divide audio clip into windows
    windows = env.open_sample_split(samp_name).to(device)  # returns a 2D array with nr windows x sample duration

    # Calculate observation based on current state for window = 0
    temp_brir = open_brir(brir_name).to(device)  # returns relevant BRIR
    temp_snd = windows[0]
    temp_snd = rms_norm(temp_snd, rms_target)  # rms normalise segment
    observation = env.convolve_sound(temp_snd, temp_brir)[0:2].to(device)  # this convolves the segment with the BRIR

    # Loop over all windows in the episode:
    for window in range(env.n_windows):

        # Select action epsilon-greedily:
        Q_vals = env.get_Q_values_from_state(observation.unsqueeze(
            0))  # this gives the state (observation) as input and returns the Q-values for all actions from policy network
        best_action = env.get_best_actions_from_Q_values(
            Q_vals)  # returns max Q-value, doesn't need to be a function, can just be replaced with argmax here
        selected_action, action_idx = env.sample_action_epsilon_greedily(best_action,
                                                                         epsilon)  # samples action based on epsilon greedy policy, could be defined differently as well

        # Execute action and observe reward:
        new_heador_az, new_heador_el, temp_DOA_az, temp_DOA_el = env.take_action(selected_action, heador_az, heador_el,
                                                                                 int(tar_azimuth),
                                                                                 int(tar_elevation))  # return position of new state
        reward = env.get_reward(new_heador_az, new_heador_el, heador_az, heador_el, tar_azimuth,
                                tar_elevation)  # calculate reward for this action

        # Update metrics
        trajectory.append([int(new_heador_az) if int(new_heador_az) <= 90 else -((-int(new_heador_az)) % 360),
                           int(new_heador_az)])  # add to metric
        total_reward += reward  # update total reward for this action

        # Set new state
        terminal = env.is_terminal(new_heador_az, new_heador_el, tar_azimuth,
                                   tar_elevation)  # first check whether state is terminal
        if not terminal:
            brir_name = env.next_state(brir_name, new_heador_az,
                                       new_heador_el)  # takes current brir name and returns new brir_name
            temp_brir = open_brir(brir_name).to(device)  # returns relevant BRIR
            temp_snd = windows[window:window + 1]
            temp_snd = rms_norm(temp_snd, rms_target)  # rms normalise segment
            new_observation = env.convolve_sound(temp_snd, temp_brir)[0:2].to(device)
        else:  # in case state is terminal
            new_observation = torch.zeros(2, length_windows).to(device)

        # Break loop if target is reached 
        if terminal:
            break

        # Go to next state:
        heador_az, heador_el = new_heador_az, new_heador_el
        observation = new_observation

    # Calculate metrics
    # raw episode length
    episode_length.append(window + 1)  # add 1 to make episode length between 1 and 20 (instead of between 0 and 19)
    # norm episode length 
    episode_length_norm.append(norm_length_episode(window + 1, int(tar_azimuth), int(tar_elevation)))
    # remaining chebyshev distance
    end_distances.append(
        chebyshev_distance(int(new_heador_az), int(new_heador_el), int(tar_azimuth), int(tar_elevation)))
    # accumulated reward
    all_rewards_raw.append(total_reward)  # raw total reward
    all_rewards_norm.append(
        norm_reward_episode(total_reward, int(tar_azimuth), int(tar_elevation), reward_optimal_choice,
                            reward_target_reached))

    # trajectories
    trajectories.append(trajectory)

    wandb.log({
        "Epoch": ep,
        "Episode Length Raw": window + 1,
        "Episode Length Norm": norm_length_episode(window + 1, int(tar_azimuth), int(tar_elevation)),
        "Remaining Distance": chebyshev_distance(int(new_heador_az), int(new_heador_el), int(tar_azimuth),
                                                 int(tar_elevation)),
        "Reward Raw": total_reward,
        "Reward Norm": norm_reward_episode(total_reward, int(tar_azimuth), int(tar_elevation), reward_optimal_choice,
                                           reward_target_reached),
    })

# Initialize variables to track metrics
print("Mean episode length:", np.mean(episode_length))
print("Mean normalized episode length:", np.mean(episode_length_norm))
print("Mean remaining distance:", np.mean(end_distances))
print("Mean reward:", np.mean(all_rewards_raw))
print("Mean normalized reward:", np.mean(all_rewards_norm))
