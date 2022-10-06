import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_states, num_inputs) -> None:
        super().__init__()

        # The inputs to the model is the number of states + number of inputs of the alu-plant
        self._num_inputs = num_states + num_inputs
        self._num_outputs = num_states

        self._model = nn.Sequential(
            nn.Linear(self._num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self._num_outputs),
        )

        # Adjust weight decay to add L2 regularization
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._model(input)
