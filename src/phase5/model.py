from __future__ import annotations

import torch
from torch import nn


class VideoPredictionModel(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], hidden_size: int) -> None:
        super().__init__()
        channels, height, width = input_shape
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        frame_feature_size = channels * height * width

        self.frame_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(frame_feature_size, hidden_size),
            nn.ReLU(),
        )
        self.temporal_model = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.frame_decoder = nn.Sequential(
            nn.Linear(hidden_size, frame_feature_size),
            nn.Sigmoid(),
        )
        self.channels = channels
        self.height = height
        self.width = width

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        return self.frame_encoder(frame)

    def encode_sequence(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels, height, width = frame_sequence.shape
        encoded_frames = self.frame_encoder(frame_sequence.view(batch_size * sequence_length, channels, height, width))
        encoded_frames = encoded_frames.view(batch_size, sequence_length, self.hidden_size)
        sequence_representation, _ = self.temporal_model(encoded_frames)
        return sequence_representation[:, -1, :]

    def predict_next_frame(self, sequence_representation: torch.Tensor) -> torch.Tensor:
        decoded = self.frame_decoder(sequence_representation)
        return decoded.view(-1, self.channels, self.height, self.width)

    def forward(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        sequence_representation = self.encode_sequence(frame_sequence)
        predicted_frame = self.predict_next_frame(sequence_representation)
        return predicted_frame


def initialize_video_model(
    input_shape: tuple[int, int, int],
    hidden_size: int,
) -> VideoPredictionModel:
    return VideoPredictionModel(input_shape=input_shape, hidden_size=hidden_size)
