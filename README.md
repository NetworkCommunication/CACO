# CACO

## Project Introduction

This paper introduces a collaborative caching and offloading (CACO) scheme.

First, to mitigate resource wastage caused by intervehicle communication interruptions, we employ Generative Adversarial Network (GAN) for trajectory prediction. This process generates a relationship matrix, predicting the stability of intervehicle link connections to assist in V2V offloading decisions.

Second, to circumvent redundant uploads and computations for recurring offloading tasks, we analyze the popularity of historical offloading tasks using the Page-Hinkley test (PHT) technique, caching frequently offloaded tasks to reduce the processing latency of offloading tasks.

Subsequently, a matching scheme based on Bloom filter for caching and offloading contents is devised.

Finally, the Deep Reinforcement Learning (DRL) algorithm is employed to train the offloading strategy.

## Environmental Dependence

The code requires python3 (>=3.6) with the development headers. The code also needs system packages as below:

tensorflow == 2.9.1,

torch == 1.11.0,

numpy == 1.22.4,

pandas == 1.4.2,

matplotlib == 3.5.2,

openpyxl == 3.0.10,

gym == 0.18.0,

bitarray,

mmh3.

If users encounter environmental problems and reference package version problems that prevent the program from running, please refer to the above installation package and corresponding version.

## How to Run

Each folder contains independent experimental configurations. To run the standard algorithm with full features (GAN + Cache + DDPG), navigate to the ddpg folder and execute run.py. For variant experiments (Ca-DDPG, Pre-DDPG, Pre-Ca-Local), run the run.py file in the corresponding folder.

## Catalog Structure

### BloomFilter

Bloom Filter implementation for efficient cache membership testing. This probabilistic data structure is used to quickly determine whether a content item exists in the vehicle cache, reducing unnecessary cache lookups and improving system efficiency.

**File description:**

* BloomFilter.py is the implementation of the Bloom Filter data structure using bitarray and mmh3 hash functions.

### GAN

We present a vehicle trajectory prediction method using Generative Adversarial Network, followed by the construction of an inter-vehicle relationship matrix. The main aim is to forecast future travel conditions for vehicles. 

**File description:**

* config.py contains path configurations and basic settings for the GAN module.
* train\_model.py is the training script for the GAN model.
* tools.py provides utility functions for data processing and model operations.
* data\_process folder contains data preprocessing scripts.
* model folder stores trained GAN models.

### DDPG

We utilize the DDPG algorithm to achieving optimal task offloading, defining the state space, action space, and reward function.

**File description:**

* env.py is the environment configuration for the DDPG algorithm, including vehicle relationships, task allocation, and reward calculation.
* network.py defines the Actor-Critic network structure of the DDPG algorithm.
* run.py is the main execution file for training and testing the DDPG agent.
* BloomFilter.py is the Bloom Filter implementation for cache management.
* Actor\_ddpg, Criticddpg, TargetActor\_ddpg, TargetCriticddpg are saved model parameters.

### Pre-Ca-Local

All tasks are processed locally on the requesting vehicle without offloading to service vehicles, serving as a baseline comparison.

**File description:**

* env.py is the environment configuration for local offloading scenario.
* network.py defines the network structure.
* run.py is the execution file.
* Model parameter files (Actor\_ddpg, Criticddpg, etc.).

### Pre-LFU-DDPG

DDPG algorithm variant with LFU mechanism. This configuration uses the LFU cache management to evaluate the impact of PHT caching on system performance.

**File description:**

* network.py defines the network structure.
* run.py is the execution file.
* Model parameter files.

### Ca-DDPG

DDPG algorithm variant without GAN-based content popularity prediction. This configuration uses alternative methods for content popularity estimation to evaluate the contribution of GAN.

**File description:**

* env.py is the environment configuration without GAN module.
* network.py defines the network structure.
* run.py is the execution file.
* BloomFilter.py for cache management.
* Model parameter files.

### contrast-Line-chart

Comparison experiments with different DDPG variants focusing on episode-based metrics. Includes comparisons between standard DDPG, local offloading, LFU cache variant, and no-GAN variant with bar chart visualizations.

**File description:**

* env\_ddpg.py, env\_ddpg\_local.py, env\_ddpg\_LFUcache.py, env\_ddpg\_nogan.py are environment configurations for different experimental variants.
* main\_episode\_alltime.py executes comparison experiments for total time cost across episodes.
* main\_episode\_average\_time.py executes comparison experiments for average time cost per episode.
* main\_episode\_rate.py executes comparison experiments for offloading rate analysis.
* network.py defines the DDPG network structure.
* BloomFilter.py for cache management.
* Model parameter files for each variant.

### contrast\_column\_chart

Comparison experiments with different DDPG variants focusing on time-based metrics with column chart visualizations. Analyzes the performance differences across different configurations.

**File description:**

* t\_alltime\_env\_ddpg.py, t\_alltime\_env\_ddpg\_local.py, t\_alltime\_env\_ddpg\_LFUcache.py, t\_alltime\_env\_ddpg\_nogan.py are environment configurations for time-based comparison experiments.
* main\_t\_alltime.py is the main execution file for time-based comparison experiments.
* network.py defines the network structure.
* Model parameter files.

## Statement

In this project, due to the different parameter settings of vehicle, task, RSU, etc., the parameters of reinforcement learning algorithm are set differently, and the reinforcement learning process varies, resulting in different experimental results. The GAN module is used for vehicle trajectory prediction. The Bloom Filter provides efficient cache membership testing. DDPG is used for offloading decisions. Multiple experimental configurations are provided to evaluate the contribution of each component (GAN, Cache, Offloading) to the overall system performance.

