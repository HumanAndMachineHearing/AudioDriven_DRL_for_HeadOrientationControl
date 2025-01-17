# Audio-Driven Deep Reinforcement Learning for Head Orientation Control in Naturalistic Environments

## Introduction

This repository provides an implementation of a reinforcement learning framework utilising deep Q-learning and a recurrent neural network for audio-driven head orientation control. The framework is described in detail in our paper: 

Ledder, W., Qin, Y., & van der Heijden, K. (2024). Audio-Driven Reinforcement Learning for Head-Orientation in Naturalistic Environments. arXiv preprint arXiv:2409.10048.

## Abstract

Although deep reinforcement learning (DRL) approaches in audio signal processing have seen substantial progress in recent years, fully audio-driven DRL for tasks such as navigation, gaze control and head-orientation control have received little attention. Yet, such audio-driven DRL approaches are highly relevant for the development of fully autonomous audio-based agents as they can be seamlessly merged with other audio-driven (DRL) approaches such as automatic speech recognition and emotion recognition. Therefore, we propose an end-to-end, audio-driven DRL framework in which we utilize deep Q-learning to develop an autonomous agent that orients towards a talker in the acoustic environment based on stereo speech recordings. Our results show that the agent learned to perform the task in a range of naturalistic acoustic environments with varying degrees of reverberation. Quantifying the degree of generalization of the proposed DRL approach across acoustic environments revealed that policies learned by an agent trained on medium or high reverb environments generalized to low reverb environments, but policies learned by an agent trained on anechoic or low reverb environments did not generalize to medium or high reverb environments. Taken together, this study demonstrates the potential of fully audio-driven DRL for tasks such as head-orientation control. Furthermore, our findings highlight the need for training strategies that enable robust generalization across acoustic environments in order to develop real-world audio-driven DRL applications.  

## Contents

### Data

Agents were trained on speech clips of 10 s duration, retrieved from the LibriSpeech ASR corpus [1]. Each episode starts with the agent's head oriented in an azimuth and elevation angle w.r.t. the talker (and also w.r.t. the room, for reverberant scenes). Starting positions were randomly assigned to each speech clip. The random sampling files show the combinations of speech clips and starting positions. The HRTFs used for simulating anechoic scenes were retrieved from the SOFA HRTF library [2]. The BRIRs used for simulating reverberant scenes were computed convolving RIRs (simulated by PyroomAcoustics [3]) with HRTFs using Binaural SDM [4].

### Requirements

The requirements can be found in the file `requirements.txt`. To make a Python virtual environment with the necessary packages, run the following commands:

```bash
git clone https://github.com/HumanAndMachineHearing/AudioDriven_DRL_for_HeadOrientationControl  # to get scripts and data
python3 -m venv venv
source venv/bin/activate  # for Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

### Code

The training script provided can be used to train the weights of the deep Q-network for environments with a given level of reverberation (anechoic, low, medium or high). The testing script can be used to test the agent on the environment it has been trained on, or on a different environment.

To run the scripts, the following commands can be used:

```bash
source venv/bin/activate  # for Windows, use: venv\Scripts\activate
python3 Models/RL_BRIR_xxx.py
```

Here, `xxx` should be replaced by `anechoic`, `lowreverb`, `midreverb` or `highreverb`.

## Citation
If you find this repository helpful in an academic setting, please cite: 

```bibtex
@article{ledder2024audio,
title={Audio-Driven Reinforcement Learning for Head-Orientation in Naturalistic Environments},
author={Ledder, Wessel and Qin, Yuzhen and van der Heijden, Kiki},
journal={arXiv preprint arXiv:2409.10048},
year={2024}
}
```

The BibTeX will be updated once the ICASSP 2025 proceedings are published.

## References

[1] Panayotov, V., Chen, G., Povey, D., and Khudanpur, S. (2015, April). Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 5206-5210). IEEE.

[2] Majdak, P., Iwaya, Y., Carpentier, T., Nicol, R., Parmentier, M., Roginska, A., ... and Noisternig, M. (2013, May). Spatially oriented format for acoustics: A data exchange format representing head-related transfer functions. In Audio Engineering Society Convention 134. Audio Engineering Society.

[3] Scheibler, R., Bezzam, E., and Dokmanić, I. (2018, April). Pyroomacoustics: A python package for audio room simulation and array processing algorithms. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 351-355). IEEE.

[4] Amengual Garí, S. V., Arend, J. M., Calamia, P. T., and Robinson, P. W. (2021). Optimizations of the spatial decomposition method for binaural reproduction. Journal of the Audio Engineering Society, 68(12), 959-976.