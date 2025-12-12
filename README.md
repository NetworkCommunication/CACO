# CACO
This paper introduces a collaborative caching and offloading (CACO) scheme. 

First, to mitigate resource wastage caused by intervehicle communication interruptions, we employ Generative Adversarial Network (GAN) for trajectory prediction. This process generates a relationship matrix, predicting the stability of intervehicle link connections to assist in V2V offloading decisions.

Second, to circumvent redundant uploads and computations for recurring offloading tasks, we analyze the popularity of historical offloading tasks using the Page-Hinkley test (PHT) technique, caching frequently offloaded tasks to reduce the processing latency of offloading tasks. 

Subsequently, a matching scheme based on Bloom filter for caching and offloading contents is devised. 

Finally, the Deep Reinforcement Learning (DRL) algorithm is employed to train the offloading strategy.
