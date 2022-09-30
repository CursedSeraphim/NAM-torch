# import iris dataset and the NAM
from sklearn.datasets import load_iris
from NAM import NAM
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
model = NAM(4, 10, 3, 2)
# define a regular neural network with dropout
# model = th.nn.Sequential(
#     th.nn.Linear(4, 10),
#     th.nn.ELU(),
#     th.nn.Dropout(0.5),
#     th.nn.Linear(10, 10),
#     th.nn.ELU(),
#     th.nn.Dropout(0.5),
#     th.nn.Linear(10, 3)
# )

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

# get the submodule feature_maps[input_feature, output_feature, value]
feature_maps = model.get_feature_maps()
print(feature_maps.shape)

# use matplotlib to create a plot of the feature maps such that there is a row for each input dimension and a column for each feature map
fig, axs = plt.subplots(4, 3, figsize=(10, 10))
for i in range(4):
    for j in range(3):
        # define an x-axis for the feature maps that corresponds to the input range of the input feature
        x = np.linspace(input_ranges[i,0], input_ranges[i,1], 100)
        # plot the feature map
        axs[i,j].plot(x, feature_maps[i,j,:])
        # use both the feature dictionary and the output class dictionary to set the title
        axs[i, j].set_title(f'{feature_dict[i]} for {output_class_dict[j]}')

# add a gridline at y=0
for i in range(4):
    for j in range(3):
        axs[i,j].axhline(0, color='black', linestyle='--')

# make sure the plots are not overlapping
plt.tight_layout()
plt.show()