from NAM import NAM2DOnly
from sklearn.datasets import load_iris
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# load the dataset
sklearn_dataset = load_iris()
X = sklearn_dataset.data
y = sklearn_dataset.target

# store the original range of the input dimensions
input_ranges = np.zeros((X.shape[1], 2))
for i in range(X.shape[1]):
    input_ranges[i,0] = np.min(X[:,i])
    input_ranges[i,1] = np.max(X[:,i])
    X[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))

# normalize the data
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# use the sklearn_dataset object to create a dictionary of which output class corresponds to which integer
output_class_dict = {i: sklearn_dataset.target_names[i] for i in range(len(sklearn_dataset.target_names))}
# use the sklearn_dataset object to create a dictionary of which input dimension corresponds to which feature
feature_dict = {i: sklearn_dataset.feature_names[i] for i in range(len(sklearn_dataset.feature_names))}

# split
train_size = int(0.8 * X.shape[0])
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# define the neural additive model
model = NAM2DOnly(4, 10, 3, 4)

# initialize the model parameters, set all seeds for reproducibility
seed = 2
th.manual_seed(seed)
np.random.seed(seed)
model.apply(model.init_weights)

# print the model
print(model)

# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=4e-3)

# define the loss function
loss_fn = F.cross_entropy

# define method for training on training data
def train(model, X, y, optimizer, loss_fn):
    # set model to training mode
    model.train()
    # convert data to tensors
    X = th.tensor(X, dtype=th.float32)
    y = th.tensor(y, dtype=th.long)
    # zero the gradients
    optimizer.zero_grad()
    # forward pass
    output = model(X)
    # compute loss
    loss = loss_fn(output, y)
    # backpropagate
    loss.backward()
    # update parameters
    optimizer.step()
    return loss.item()

# define method for evaluating on test data
def evaluate(model, X, y, loss_fn):
    # set model to evaluation mode
    model.eval()
    # convert data to tensors
    X = th.tensor(X, dtype=th.float32)
    y = th.tensor(y, dtype=th.long)
    # forward pass
    output = model(X)
    # compute loss
    loss = loss_fn(output, y)
    # compute accuracy
    accuracy = (output.argmax(dim=1) == y).sum().item() / y.shape[0]
    return loss.item(), accuracy

# train epochs
for epoch in range(1000):
    # train
    train_loss = train(model, X_train, y_train, optimizer, loss_fn)
    # evaluate
    test_loss, test_accuracy = evaluate(model, X_test, y_test, loss_fn)
    # print
    print(f'Epoch {epoch+1}: train loss = {train_loss:.4f}, test loss = {test_loss:.4f}, test accuracy = {test_accuracy:.4f}')

# print the final test accuracy
print(f'Final test accuracy: {test_accuracy:.4f}')

# get the submodule feature_maps
resolution = 5
feature_maps_1D, feature_maps_2D = model.get_feature_maps(resolution=resolution)
print(feature_maps_1D.shape)
print(feature_maps_2D.shape)

# visualize 1D feature_maps
fig, axs = plt.subplots(4, 3, figsize=(10, 10))
for i in range(4):
    for j in range(3):
        x = np.linspace(input_ranges[i, 0], input_ranges[i, 1], resolution)
        axs[i, j].plot(x, feature_maps_1D[i, :, j])
        axs[i, j].set_title(f'{feature_dict[i]} for {output_class_dict[j]}')
        axs[i, j].axhline(0, color='black', linestyle='--')

plt.tight_layout()
plt.show()

# visualize 2D feature_maps
pair_indices = list(combinations(range(4), 2))
num_pairs = len(pair_indices)

fig, axs = plt.subplots(num_pairs, 3, figsize=(12, 4 * num_pairs), squeeze=False)
for pair_idx, (i, j) in enumerate(pair_indices):
    for output_feature in range(3):
        x = np.linspace(input_ranges[i, 0], input_ranges[i, 1], resolution)
        y = np.linspace(input_ranges[j, 0], input_ranges[j, 1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = feature_maps_2D[pair_idx, :, :, output_feature].T

        im = axs[pair_idx, output_feature].contourf(X, Y, Z, cmap='viridis', levels=20)
        axs[pair_idx, output_feature].set_title(f'{feature_dict[i]} and {feature_dict[j]} for {output_class_dict[output_feature]}')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.6)
plt.show()
