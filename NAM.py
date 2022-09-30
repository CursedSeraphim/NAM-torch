import torch as th
import numpy as np

# define the Neural Additive Model
class NAM(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        # for each input dimension, we have an individual network with num_layers layers
        # the output of the overall network is the sum of the outputs of the individual networks, fed into a softmax for classification
        super(NAM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.submodules = th.nn.ModuleList()
        # initialize the submodules and make sure they accept input dimension = 1 for the first layer and hidden_dim for all other layers. they should have num_layers layers and output_dim output dimensions
        # also use dropout with p=0.5
        for i in range(input_dim):
            # initialize the submodule
            submodule = th.nn.Sequential()
            for l in range(num_layers):
                if l == 0:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(1, hidden_dim))
                else:
                    submodule.add_module(f"linear_{l}", th.nn.Linear(hidden_dim, hidden_dim))
                submodule.add_module(f"ELU_{l}", th.nn.ELU())
                submodule.add_module(f"dropout_{l}", th.nn.Dropout(0.5))
            # each subnetwork has a final linear layer to output the final output
            submodule.add_module(f"linear_{num_layers}", th.nn.Linear(hidden_dim, output_dim))
            # add the submodule to the list of submodules
            self.submodules.append(submodule)
                        
    
    def forward(self, x):
        """
        The forward pass passes each input dimension through the corresponding submodule and sums over their outputs. The output is then fed into a softmax for classification.
        """
        # initialize the output
        output = th.zeros(x.shape[0], self.output_dim)
        # for each input dimension, pass it through the corresponding submodule and add the output to the overall output
        for i in range(self.input_dim):
            output += self.submodules[i](x[:,i].unsqueeze(1))
        # return the softmax of the output
        return th.nn.functional.softmax(output, dim=1)

    def init_weights(self, m):
        if type(m) == th.nn.Linear:
            th.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # output what each submodule predicts for each input between 0 and 1 for a given resolution
    def get_feature_maps(self, resolution=100):
        # initialize the output
        output = th.zeros(resolution, self.input_dim, self.output_dim)
        # for each input dimension, pass it through the corresponding submodule and add the output to the overall output
        for i in range(self.input_dim):
            for j in range(resolution):
                output[j,i] = self.submodules[i](th.tensor([[j/resolution]]))
        # return output as numpy array
        return np.moveaxis(output.detach().numpy(), 0, -1)