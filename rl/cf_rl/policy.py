"""Small CNN feature extractor for the 64x64 stacked grayscale camera.

Used as ``CnnPolicy``'s ``features_extractor_class``. Stable-baselines3
normalizes uint8 images to 0..1 before this module runs.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """Three stride-2 convolutions, then a 64-d projection.

    Sized for a 64x64 input. NatureCNN also fits that resolution; this one
    is the network ``train.py`` actually uses for the image-only policy.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        channels = int(observation_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            flat = self.cnn(sample)
        self.head = nn.Sequential(nn.Linear(flat.shape[1], features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(observations))
