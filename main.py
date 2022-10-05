import pickle
import numpy as np
import torch
import torch.nn as nn


from dnn import Predictor


def main():
    num_epochs = 20

    with open("PBM_simulation_data.pickle", "rb") as handle:
        data = pickle.load(handle)

    pbm_states = 8
    pbm_inputs = 5

    num_samples = data.shape[0]
    num_train = num_samples // 2
    num_validation = num_samples // 4

    x_train = data[:num_train, :]
    x_val = data[num_train : num_train + num_validation, :]
    x_test = data[num_train + num_validation : -1, :]

    y_train = data[1 : num_train + 1, :, :pbm_states]  # TODO: LOOK AT DIM
    y_val = data[num_train + 1 : num_train + num_validation + 1, :pbm_states]
    y_test = data[num_train + num_validation + 1 :, :pbm_states]

    model = Predictor(pbm_states, pbm_inputs)

    # Debugging: Validate sizes
    print(len(x_train), len(y_train))
    print(len(x_val), len(y_val))
    print(len(x_test), len(y_test))
    print()

    x_train = torch.tensor(x_train, dtype=torch.float).to(model.device)
    y_train = torch.tensor(y_train, dtype=torch.float).to(model.device)

    for epoch in range(num_epochs):
        pred = model.forward(x_train)
        print(pred.shape)
        print(y_train.shape)
        loss = nn.functional.mse_loss(pred, y_train)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        print(loss)


if __name__ == "__main__":
    main()
