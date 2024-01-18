"""
    replay_buffer.py

    Necessary component to stabilize Deep Deterministic Policy Gradient-based Reinforcement Learner agent;
    the agent learns by sampling batches from its 'recent memory' (this thing).
    
    Implemented as a contents-agnostic ring buffer.
"""

import random
import sys


class ReplayBuffer():
    """
        Replay Buffer memory for DDPG RL agents.
        Initialize with fixed size (oldest observations will be overwritten).
        Offers storing, sampling, and clearing functionality; items kept in the buffer must be tuples of
        uniform size, as the sampling functionality will batch tuple elements together by index.
        The buffer doesn't check this, for speed.
    """
    
    def __init__(self, size):
        self.backing_size = size
        self.head = 0
        self.len = 0
        self.backing = [None for i in range(self.backing_size)]

    def __len__(self):
        return self.len


    def clear(self):
        """ Clears the buffer (doesn't actually erase anything, just moves the control indices). """
        self.head = 0
        self.len = 0


    def store(self, item : tuple):
        """ Store an item in the buffer, evicting the least recent item if the buffer is full. """

        if(self.len == 0):
            self.backing[self.head] = item
            self.len = 1

        else:
            tail = (self.head + self.len) % self.backing_size
            self.backing[tail] = item
            if tail == self.head:
                # Buffer full; evict last item
                self.head = (self.head + 1) % self.backing_size
            else:
                # Space remains; increase apparent length
                self.len += 1


    def sample_into_batches(self, n : int):
        """ Draws n items from the replay buffer, and groups each item's elements together by index, returning a tuple of item lists. """

        assert self.len > 0, "Cannot sample from an empty replay buffer"
        
        try:
            sample_indices = random.sample(range(self.len), n)

        except ValueError:
            # Not enough data to sample without replacement
            print(f"Warning: attempted to sample {n} items from replay memory of length {self.len}. Sampling with replacement!", file=sys.stderr)
            sample_indices = [random.randrange(self.len) for i in range(n)]


        num_batches = len(self.backing[self.head])
        output = [[] for i in range(num_batches)]

        for index in sample_indices:
            # For every sampled observation

            item = self.backing[(self.head + index) % self.backing_size]

            for batch_index in range(num_batches):
                # For every element in an observation
                output[batch_index].append(item[batch_index])

        return output

