from __future__ import annotations

import torch
from torch import nn


class AudioSequenceModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.lstm(sequence)
        predicted_next_sequence = self.output_layer(hidden_states)
        return predicted_next_sequence


def initialize_audio_model(input_size: int, hidden_size: int, output_size: int) -> AudioSequenceModel:
    model = AudioSequenceModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    return model
