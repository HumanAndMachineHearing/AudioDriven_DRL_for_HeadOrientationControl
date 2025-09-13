# ------------------------------------------------------------
# Import packages and libraries
# ------------------------------------------------------------

import numpy as np
import wandb
import torch
import torchaudio
import sofa
import torch.nn as nn
from tqdm import tqdm
import os
import random

#%%
# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------
dir_random_sampling_file = "/fldr_randomsampling"
dir_data = "/fldr_data"
dir_hrtf = '/fldr_hrtf'

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
gamma = 0.8  # discount factor (was 1 in Wessel's thesis, based on Yuzhen's advice: change to 0.99 later on).
learning_rate = 0.001  # was 0.00025
batch_size = 1024
reward_optimal_choice = 0.1
reward_suboptimal_choice = 0  # try -0.1
reward_bad_choice = -0.2
reward_target_reached = 1
memory_capacity = 5000  # number of experiences to store per location said 20000 first?
rms_target = 10  # RMS scaling target for data normalisation

# Locations
az_angles = ["090", "075", "060", "045", "030", "015", "000", "345", "330", "315", "300", "285", "270"]  # head orientation
DOA_az_angles = ["-90", "-75", "-60", "-45", "-30", "-15", "000", "015", "030", "045", "060", "075", "090"]
el_angles = ["-45", "-20", "000", "020", "045"]
DOA_el_angles = ["-40", "-20", "000", "020", "040"]

all_locs_for_buffer = list()  # initialise empty list
for i in range(len(DOA_el_angles)):
    for j in range(len(DOA_az_angles)):
        all_locs_for_buffer.append([DOA_el_angles[i], DOA_az_angles[j]])

# Read HRTF
HRTF = sofa.Database.open(dir_hrtf)
positions = np.round(HRTF.Source.Position.get_values(system="spherical"),
                     1)  # SOFA frame of reference: 0 = straight ahead, 90 = left, 270 = right!


#%%
# ------------------------------------------------------------
# Functions miscellaneous
# ------------------------------------------------------------
def initial_states_to_list(random_sampling_file):
    # this function generates list "initial states" which contains for each line in the random sampling file the azimuth, elevation and file name of the speaker (why?)

    initial_states = list()
    f = open(random_sampling_file, "r")
    for line in f:
        az, el, file = line.split(" ")
        initial_states.append((az, el, file.removesuffix("\n")))
    f.close()

    return initial_states


def chebyshev_distance(hor_az, hor_el):
    # Function to calculate Chebyshev distance between current orientation and target orientation (az = 0, el = 0)

    # retrieve azimuth and elevation deviation
    cur_az_deviation = calc_doa_az(int("000"), int(hor_az))
    cur_el_deviation = calc_doa_el(int("000"), int(hor_el))
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


def calc_doa_az(target_hor, current_hor):
    # specify for flipping around, if heador is to the right of target, positive difference; if heador is to the left of target, negative difference
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
# Memory buffer
# ------------------------------------------------------------

class BalancedMemoryBuffer():
    def __init__(self, batsize, locs, capacity, dev):
        self.device = dev

        # Assign values to object properties    
        self.buffer_length = capacity  # numpber of experiences per location combination
        self.num_classes = len(locs)  # number of possible locations
        self.locations = locs  # all location combinations, used for indexing where to assign in the buffer
        self.batch_size = batsize  # batch size
        self.exp_counter = np.zeros((self.num_classes))  # this is a counter that tracks at which experience to store
        self.sam_per_loc = int(np.ceil(
            self.batch_size / self.num_classes))  # Determine number of samples to take per location when taking a sample from buffer
        self.suf_exp = torch.zeros(self.num_classes,
                                   1)  # keeps track of whether memory buffere has been filled at least once for this location

        # initialise an empty buffer
        self.state = torch.empty((self.num_classes, self.buffer_length, 2, 4000)).to(self.device)  # states
        self.action = torch.empty((self.num_classes, self.buffer_length, 1), dtype=torch.int).to(self.device)  # actions
        self.next_state = torch.empty((self.num_classes, self.buffer_length, 2, 4000)).to(self.device)  # states+1
        self.reward = torch.empty((self.num_classes, self.buffer_length, 1)).to(self.device)  # reward
        self.state_terminal = torch.empty((self.num_classes, self.buffer_length, 1), dtype=torch.bool).to(
            self.device)  # mask for terminal states, boolean

    def push_to_buffer(self, s, a, s_prime, r, t, azi, ele):
        """
        Function that adds a <s,a,s',r> tuple into the buffer for a specific location.
        s: state
        a: action
        s_prime: new state
        r: reward
        t: terminal state
        az: azimuth angle
        el: elevation angle
        """

        # First index = location to assign to buffer, second index = experience number to determine where to assign to buffer
        self.state[self.locations.index([ele, azi]), self.exp_counter, :, :] = s
        self.action[self.locations.index([ele, azi]), self.exp_counter] = torch.tensor([a], dtype=torch.int)
        self.next_state[self.locations.index([ele, azi]), self.exp_counter, :, :] = s_prime
        self.reward[self.locations.index([ele, azi]), self.exp_counter] = torch.tensor([r], dtype=torch.float)
        self.state_terminal[self.locations.index([ele, azi]), self.exp_counter] = torch.tensor([t],
                                                                                               dtype=torch.bool)  # this is a mask of which states are terminal and which are not, which is used for the optimization loop

        # Update experience counter for this particular location  
        if self.exp_counter[self.locations.index([ele, azi])] < self.buffer_length - 1:
            self.exp_counter[self.locations.index([ele, azi])] += 1
        else:
            self.suf_exp[self.locations.index([ele, azi])] = 1  # set to 1 for this location
            self.exp_counter[
                self.locations.index([ele, azi])] = 0  # set back to zero in case maximum capacity is reached

    def sample_from_buffer(self):  # this one needs to be adapted based on the correct sampling
        """
        Function that returns random samples from the buffer with equal representation from each class.
        """

        # Sample from buffer
        if all(i >= self.sam_per_loc for i in
               self.exp_counter):  # check whether there are sufficient experiences for each location
            # initialise an empty sample of size batch_size/num_locations, this is used later when sampling from buffer
            sample_state = torch.empty((self.num_classes * self.sam_per_loc, 2, 4000)).to(self.device)  # states
            sample_action = torch.empty((self.num_classes * self.sam_per_loc, 1), dtype=torch.int).to(
                self.device)  # actions
            sample_next_state = torch.empty((self.num_classes * self.sam_per_loc, 2, 4000)).to(self.device)  # states+1
            sample_reward = torch.empty((self.num_classes * self.sam_per_loc, 1)).to(self.device)  # reward
            sample_state_terminal = torch.empty((self.num_classes * self.sam_per_loc, 1), dtype=torch.bool).to(
                self.device)  # mask for terminal states, boolean

            for j in range(self.num_classes):  # cycle through all locations
                sample_idx = random.sample(range(int(self.exp_counter[j])),
                                           self.sam_per_loc)  # take a random sample from all experiences stored so far
                # add to sample
                sample_state[j * self.sam_per_loc:(j + 1) * self.sam_per_loc, :, :] = self.state[j, sample_idx, :, :]
                sample_action[j * self.sam_per_loc:(j + 1) * self.sam_per_loc] = self.action[j, sample_idx]
                sample_next_state[j * self.sam_per_loc:(j + 1) * self.sam_per_loc, :, :] = self.next_state[j,
                                                                                           sample_idx, :, :]
                sample_reward[j * self.sam_per_loc:(j + 1) * self.sam_per_loc] = self.reward[j, sample_idx]
                sample_state_terminal[j * self.sam_per_loc:(j + 1) * self.sam_per_loc] = self.state_terminal[
                    j, sample_idx]

            # The samples that are exceeding batch size are discarded, shuffle first for random discarding 
            if np.shape(sample_state)[0] > self.batch_size:
                retain_idx = random.sample(range(np.shape(sample_state)[0]),
                                           self.batch_size)  # sample random indices corresponding to batch size for retaining
                sample_state = sample_state[retain_idx]
                sample_action = sample_action[retain_idx]
                sample_next_state = sample_next_state[retain_idx]
                sample_reward = sample_reward[retain_idx]
                sample_state_terminal = sample_state_terminal[retain_idx]

        else:
            # initialise an empty sample of size total number of experiences, this is used later when sampling from buffer
            sample_state = torch.empty((int(np.sum(self.exp_counter)), 2, 4000)).to(self.device)  # states
            sample_action = torch.empty((int(np.sum(self.exp_counter)), 1), dtype=torch.int).to(self.device)  # actions
            sample_next_state = torch.empty((int(np.sum(self.exp_counter)), 2, 4000)).to(self.device)  # states+1
            sample_reward = torch.empty((int(np.sum(self.exp_counter)), 1)).to(self.device)  # reward
            sample_state_terminal = torch.empty((int(np.sum(self.exp_counter)), 1), dtype=torch.bool).to(
                self.device)  # mask for terminal states, boolean

            cnt = 0  # initialise counter to add experiences

            for j in range(self.num_classes):  # cycle through all locations
                sample_state[cnt:cnt + int(self.exp_counter[j]), :, :] = self.state[j, :int(self.exp_counter[j]), :, :]
                sample_action[cnt:cnt + int(self.exp_counter[j])] = self.action[j, :int(self.exp_counter[j])]
                sample_next_state[cnt:cnt + int(self.exp_counter[j]), :, :] = self.next_state[j,
                                                                              :int(self.exp_counter[j]), :, :]
                sample_reward[cnt:cnt + int(self.exp_counter[j])] = self.reward[j, :int(self.exp_counter[j])]
                sample_state_terminal[cnt:cnt + int(self.exp_counter[j])] = self.state_terminal[j,
                                                                            :int(self.exp_counter[j])]
                cnt = cnt + int(self.exp_counter[j])

            # The samples that are exceeding batch size are discarded, shuffle first for random discarding 
            if np.shape(sample_state)[0] > self.batch_size:
                retain_idx = random.sample(range(np.shape(sample_state)[0]),
                                           self.batch_size)  # sample random indices corresponding to batch size for retaining
                sample_state = sample_state[retain_idx]
                sample_action = sample_action[retain_idx]
                sample_next_state = sample_next_state[retain_idx]
                sample_reward = sample_reward[retain_idx]
                sample_state_terminal = sample_state_terminal[retain_idx]

                # should they be randomly shuffled to before returning? yes
        perm_indices = torch.randperm(np.shape(sample_state)[0]).to(self.device)
        sample_state = sample_state[perm_indices]
        sample_action = sample_action[perm_indices]
        sample_next_state = sample_next_state[perm_indices]
        sample_reward = sample_reward[perm_indices]
        sample_state_terminal = sample_state_terminal[perm_indices]

        return sample_state, sample_action, sample_next_state, sample_reward, sample_state_terminal


#%%
# ------------------------------------------------------------
# Environment functions
# ------------------------------------------------------------
def open_hrtf(az, el, pos):
    # Returns the hrtf corresponding to that specific angle down sampled to 16000 Hz
    elevation = el  # specify elevation angle to be extracted
    azimuth = az  # specify azimuth angle to be extracted

    az_index = np.argwhere(pos[:, 0] == azimuth)
    el_index = np.argwhere(pos[:, 1] == elevation)

    loc_index = np.intersect1d(az_index, el_index)

    measurement = int(loc_index[0])  # complicated way of turning numpy array into a single integer for plotting
    emitter = 0

    t = np.arange(0, HRTF.Dimensions.N) * HRTF.Data.SamplingRate.get_values(indices={"M": measurement})  # time axis
    h = np.zeros((2, len(t)))
    h[0] = HRTF.Data.IR.get_values(indices={"M": measurement, "R": 0, "E": emitter})  # left channel
    h[1] = HRTF.Data.IR.get_values(indices={"M": measurement, "R": 1, "E": emitter})  # right channel

    # downsample to 16000 kHz, which is the original sampling rate of the audio samples
    current_fs = 48000
    dwnsmpl_fs = 8000
    stepsize = int(np.floor(current_fs / dwnsmpl_fs))

    h = h[:, ::stepsize]

    return h


class environment():

    def __init__(self, fs, gamma, lr, batsize, n_windows, length_windows, az_angles, el_angles, random_sampling_file,
                 target_network, policy_network, memory_buffer, n_actions, actions, device, rew_opt_ch, rew_subopt_ch,
                 rew_bad_ch):
        self.device = device
        self.fs = fs
        self.n_windows = n_windows
        self.length_windows = length_windows
        self.az_angles = az_angles
        self.el_angles = el_angles
        self.DOA_az_angles = DOA_az_angles
        self.DOA_el_angles = DOA_el_angles
        self.init_states = initial_states_to_list(random_sampling_file)
        self.target_network = target_network.to(device)
        self.policy_network = policy_network.to(device)
        self.memory_buffer = memory_buffer
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
        sample = sample[:, ::2]  # downsample sound to 8 kHz
        return sample.reshape((self.n_windows, -1))

    def convolve_sound(self, window, hrtf):
        """
        Convolves the HRTF with the window, and cuts result at the size of the window length.
        window: the window of the sample that should be convolved.

        """
        return torchaudio.functional.convolve(window.repeat([2, 1]), hrtf)[:, :self.length_windows]
        # return torchaudio.functional.convolve(windows[0].reshape(1,-1).repeat([2,1]), brir)[:,:self.length_windows]

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
        epsilon: the greediness parameter.
        Returns: the actions [az,el].
        """
        p_best_action = 1 - epsilon + epsilon / self.n_actions
        p_action = epsilon / self.n_actions
        probability_table = np.full(self.n_actions, p_action)  # assigns the probability of each action
        probability_table[best_action] = p_best_action  # assigns higher probability to best action
        index = np.random.choice(np.arange(self.n_actions),
                                 p=probability_table)  # randomly selects an action based on the probabilities in the table
        return self.action_space[index], index

    def take_action(self, current_action, az, el):
        """
        Finds the next head orientation (azimuth and elevation) after taking the action.
        actions: a tensor with two values for the azimuth and elevation angles.
        Returns: the new head orientation (azimuth, elevation), after taking the action, as well as the new DOA.
        """

        # retrieve DOAs and convert to string
        cur_az_deviation = str(calc_doa_az(int("000"), int(az)))
        cur_el_deviation = str(calc_doa_el(int("000"), int(el)))
        # make sure name consists of three characters
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
        current_az_angle_idx = self.az_angles.index(az)  # retrieve current head orientation
        current_el_angle_idx = self.el_angles.index(el)

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
        if int(el) == 45:
            hor_el = 40
        elif int(el) == -45:
            hor_el = -40

        # Calculate new head orientations and DOAs
        # head orientation goes in the opposite direction of DOA movement, therefore - elmov for current_el_angle_idx but + el move for current DOA_el_angle_idx
        new_hor_el = self.el_angles[current_el_angle_idx - el_mov]
        new_hor_az = self.az_angles[current_az_angle_idx + az_mov]
        new_DOA_az = current_DOA_az_angle_idx + az_mov  #this is an index
        new_DOA_el = current_DOA_el_angle_idx + el_mov  # this is an index

        return new_hor_az, new_hor_el, new_DOA_az, new_DOA_el

    def get_reward(self, hor_az_selectedaction, hor_el_selectedaction, old_hor_az, old_hor_el):
        # get_reward(self, new_hor_az, new_hor_el, old_hor_az, old_hor_el, DOA_az_start, DOA_el_start):
        """
        Gives reward based on the orientation deviation
        """
        # calculate DOA of selected action
        cur_az_deviation = str(calc_doa_az(int("000"), int(hor_az_selectedaction)))
        cur_el_deviation = str(calc_doa_el(int("000"), int(hor_el_selectedaction)))
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
        old_az_deviation = str(calc_doa_az(int("000"), int(old_hor_az)))
        old_el_deviation = str(calc_doa_el(int("000"), int(old_hor_el)))
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

        # if new_hor_az == DOA_az_start and new_hor_el == DOA_el_start:
        #     return 1
        else:  # if target is not reached, calculate distances for all possible actions to find whether it is optimal or suboptimal action
            dists = np.zeros(n_actions)
            for i, action in enumerate(
                    self.action_space):  # cycle through all possible actions to compute distance for each action
                temp_hor_az, temp_hor_el, temp_doa_az, temp_doa_el = self.take_action(action, old_hor_az,
                                                                                      old_hor_el)  # retrieve new heador (az, el) for this particular action
                # dists[i] = np.sqrt(int(self.DOA_az_angles[temp_doa_az])**2 + int(self.DOA_el_angles[temp_doa_el])**2)
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

    def is_terminal(self, new_hor_az, new_hor_el):
        """
        Returns a boolean testing whether the agent has reached the target (and thus is terminal)
        """
        return new_hor_az == "000" and new_hor_el == "000"


    def append_buffer(self, observation, action_indices, new_observation, reward, terminal, azi, ele):
        """
        Appends an observation into the buffer.
        observation: state s_t
        action_indices: action a_t
        new_observation: new state s_(t+1)
        reward: r
        terminal: boolean t
        az: azimuth angle of new state
        el: elevation angle of new state
        """
        self.memory_buffer.push_to_buffer(observation, action_indices, new_observation, reward, terminal, azi, ele)

    def optimize_model(self):
        """
        Trains the policy network.
        returns: the loss as computed with the Huber loss function.
        """
        if np.sum(
                self.memory_buffer.exp_counter) < self.batch_size:  # while memory does not have enough samples), no training occurs and the training loop is exited.
            return

        # Sample from replay memory
        s_batch, a_batch, s1_batch, r_batch, t_batch = self.memory_buffer.sample_from_buffer()

        a_batch = a_batch.type(torch.int64)  # change data type

        s_batch = s_batch.to(device)  # these are the states
        a_batch = a_batch.to(device)  # these are the actions
        s1_batch = s1_batch.to(device)  # these are states s_{t+1}
        r_batch = r_batch.to(device)  # these are the rewards
        t_batch = t_batch.to(device)  # this is a mask of which states are terminal and which are not

        # Compute Q-values for the samples from the buffer, i.e. retrieve from policy network 
        Q_vals = self.policy_network(s_batch).to(device)
        # Select columns of actions taken --> These are the actions which would have been taken for each batch state according to policy_net (this comment is from the pytorch tutorial)
        Q_vals = Q_vals.gather(1, a_batch).squeeze(1)

        # Compute expected action values V(s_{t+1}) for all next states (irrespective of whether this was a terminal state or not, the terminal states are set to 0 later on). 
        with torch.no_grad():
            Q1_vals = self.target_network(s1_batch).max(1)[0].detach().to(device)

        # Assign the calculated values as expected values except when state was terminal, in that case state value = 0 (determined based on t_batch, which acts as a mask here) 
        expected_Q_vals = r_batch.squeeze(1) + self.gamma * Q1_vals * (~t_batch.squeeze(1))

        # Compute loss 
        loss = env.criterion(Q_vals, expected_Q_vals)  # this should be changed to Huber loss

        # Optimize model
        self.optimizer.zero_grad()  # clears the old gradients from the previous optimization step
        loss.backward()  # computes the gradients of the loss with respect to the parameters of the models (i.e. performs backprop)
        # gradient clipping could be added in between here if needed
        self.optimizer.step()  # updates the model parameters using the gradients computed in the previous step

        return loss.item()


#%%
# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def norm_length_episode(ep_length, start_az, start_el):
    # this function calculates the length per episode normalised for the starting distance, as episodes can be closer or further away from the target

    # calculate starting distance (i.e. shortest path)
    start_dist = chebyshev_distance(start_az, start_el)
    # normalized length per episode = episode length / starting distance where a length of 1 = shortest possible path
    norm_length = np.round(ep_length / start_dist, 2)

    return norm_length


def norm_reward_episode(ep_reward, start_az, start_el, reward_opt_ch, reward_tar_reach):
    # calculate starting distance (i.e. shortest path)
    start_dist = chebyshev_distance(start_az, start_el)
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
torch.manual_seed(42)
policy_network = DQN(n_observations, n_actions).to(device)
torch.manual_seed(42)
target_network = DQN(n_observations, n_actions).to(device)

# Randomly initiliasze weights with state_dict() and copy from policy network to target network to ensure that weights are identical
target_network.load_state_dict(policy_network.state_dict())

memory_buffer = BalancedMemoryBuffer(batch_size, all_locs_for_buffer, memory_capacity, device)
#memory_buffer = MemoryBuffer(batch_size, memory_capacity)
#%%
# ------------------------------------------------------------
# Make environment
# ------------------------------------------------------------

fs = 8000  # set sampling rate
n_windows = 20  # set the number of windows
length_windows = 4000

random_sampling_file = os.path.join(dir_random_sampling_file, "random_sampling_file_traindata_hrtf.txt")

env = environment(fs, gamma, learning_rate, batch_size, n_windows, length_windows, az_angles, el_angles,
                  random_sampling_file, target_network, policy_network, memory_buffer, n_actions, actions, device,
                  reward_optimal_choice, reward_suboptimal_choice, reward_bad_choice)
#%%
# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
# Specifications
episodes = 76800  #set maximum number of episodes, shouldn't this be done with early stopping? how does that work for RL?
epsilon_episodes = 30000
epsilon_zero = 0.2  # starting value of epsilon (exploration rate)
epsilon = epsilon_zero  # original assignment
# tau = 0.005 # update rate of the network in case of soft update, currently not in use because a hard update is performed
# targnet_update_freq = 100 # number of episodes after which the target network should be updated. This is currently set to 240, but different values should be explored
tau = 0.00005  # update rate of the network in case of soft update, currently not in use because a hard update is performed
targnet_update_freq = 1  # number of episodes after which the target network should be updated. This is currently set to 240, but different values should be explored

# Initialize variables to track metrics 
episode_length = list()  #
episode_length_norm = list()
all_loss = list()  #
end_distances = list()  #
trajectories = list()  #
all_rewards_raw = list()
all_rewards_norm = list()  # list to store normalized rewards
all_epsilon = list()
DOA_in_room_heador_az0el0 = list()  # DOA of sound source at start of episode

# Set parameters for WandB logging
wandb.init(project="RL_final_runs")  # set the wandb project where this run will be logged
projectname = 'Snellius_HRTF_wRMSnorm_anechoic_4lyr_MemCap' + str(memory_capacity) + '_Eps' + str(
    epsilon_zero) + '_EpsEpisodes' + str(epsilon_episodes) + '_BatSi_' + str(batch_size) + '_Disc' + str(
    gamma) + '_TargNetFreq' + str(targnet_update_freq) + '_RewOpt' + str(reward_optimal_choice) + '_RewSOpt' + str(
    reward_suboptimal_choice) + '_RewBad' + str(reward_bad_choice) + '_LR' + str(learning_rate) + '_Tau' + str(tau)
wandb.run.name = projectname

wandb.config = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "architecture": "RNN",
    "dataset": "Wessel_RNN_RL_data",
    "epochs": episodes,
}

# Loop through every episode 
for ep in tqdm(range(episodes)):

    # Initialize variables for this episode
    trajectory = list()  # initialize trajectory
    total_reward = 0  # intialize total reward calculation for each episode

    # Initialize environment 
    az, el, samp_name = env.get_initial_state(ep)  # extract starting position (az, el) and audio name from list
    trajectory.append([int(az) if int(az) <= 90 else -((-int(az)) % 360),
                       int(el)])  # converts the distance to -90 (right) - 0 - +90 (left) rather than sticking to the SOFA convention. Wessel: done for plotting convenience
    az_start = az  # store starting values of azimuth and elevation
    el_start = el  # store starting values of azimuth and elevation

    # Divide audio clip into windows
    windows = env.open_sample_split(samp_name).to(device)  # returns a 2D array with nr windows x sample duration

    # Calculate observation based on current state for window = 0
    hrtf_temp = torch.tensor(open_hrtf(int(az), int(el), positions), dtype=torch.float,
                             device=device)  # returns hrtf at 16 kHz resolution
    observation = env.convolve_sound(windows[0:1], hrtf_temp)[0:2].to(
        device)  # this computes the spatialised sound, i.e. the observation of the state

    # Loop over all windows in the episode:
    for window in range(env.n_windows):

        # Select action epsilon-greedily:
        Q_vals = env.get_Q_values_from_state(observation.unsqueeze(
            0))  # this gives the state (observation) as input and returns the Q-values for all actions from policy network
        best_action = env.get_best_actions_from_Q_values(
            Q_vals)  # returns max Q-value, doesn't need to be a function, can just be replaced with argmax here
        actions, action_idx = env.sample_action_epsilon_greedily(best_action,
                                                                 epsilon)  # samples action based on epsilon greedy policy, could be defined differently as well

        # Execute action and observe reward:
        new_az, new_el, temp_DOA_az, temp_DOA_el = env.take_action(actions, az, el)  # return position of new state
        reward = env.get_reward(new_az, new_el, az, el)  # calculate reward for this action

        # Update metrics
        trajectory.append([int(new_az) if int(new_az) <= 90 else -((-int(new_az)) % 360), int(new_el)])  # add to metric
        total_reward += reward  # update total reward for this action

        # Set new state:
        terminal = env.is_terminal(new_az, new_el)
        if not terminal:
            hrtf_temp = torch.tensor(open_hrtf(int(new_az), int(new_el), positions), dtype=torch.float,
                                     device=device)  # returns hrtf at 16 kHz resolution
            temp_snd = windows[window:window + 1]
            temp_snd = rms_norm(temp_snd, rms_target)
            new_observation = env.convolve_sound(temp_snd, hrtf_temp)[0:2].to(device)
        else:
            new_observation = torch.zeros(2, length_windows).to(
                device)  ### Why would you store zeros in your replay memory when the episode has ended? THis should be None?

        # Store into memory:
        #env.append_buffer(observation, action_idx, new_observation, reward, terminal,az,el)
        env.append_buffer(observation, action_idx, new_observation, reward, terminal, env.DOA_az_angles[temp_DOA_az],
                          env.DOA_el_angles[temp_DOA_el])

        # Go to next state:
        az, el = new_az, new_el
        observation = new_observation

        # Break loop if target is reached
        if terminal:
            break

    # Optimize model after each episode 
    loss = env.optimize_model()  # calculate Huber loss --> currently this is done with MSE loss, needs to be updated to MSE loss

    # Calculate metrics
    # raw episode length
    episode_length.append(window + 1)  # add 1 to make episode length between 1 and 20 (instead of between 0 and 19)
    # norm episode length 
    episode_length_norm.append(norm_length_episode(window + 1, int(az_start), int(el_start)))
    # remaining chebyshev distance
    end_distances.append(chebyshev_distance(int(az), int(el)))
    # accumulated reward
    all_rewards_raw.append(total_reward)  # raw total reward
    all_rewards_norm.append(
        norm_reward_episode(total_reward, int(az_start), int(el_start), reward_optimal_choice, reward_target_reached))
    # epsilon
    all_epsilon.append(epsilon)
    # loss
    all_loss.append(loss)
    # trajectories
    trajectories.append(trajectory)

    # Update epsilon, linear decrease
    if epsilon > epsilon_zero / epsilon_episodes:
        epsilon -= (epsilon_zero / epsilon_episodes)
    else:
        epsilon = 0

    wandb.log({
        "Epoch": ep,
        "Train Loss": loss,
        "Episode Length Raw": window + 1,
        "Episode Length Norm": norm_length_episode(window + 1, int(az_start), int(el_start)),
        "Remaining Distance": chebyshev_distance(int(az), int(el)),
        "Reward Raw": total_reward,
        "Reward Norm": norm_reward_episode(total_reward, int(az_start), int(el_start), reward_optimal_choice,
                                           reward_target_reached),
        "Epsilon": epsilon
    })

    # soft update target network
    for target_param, policy_param in zip(env.target_network.parameters(), env.policy_network.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

torch.save(env.policy_network.state_dict(), projectname + 'final_episode.pth')
