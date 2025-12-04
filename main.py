from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Parameters
N = 1000  # Number of points to sample
x_min, x_max = -4, 4
y_min, y_max = -4, 4
resolution = 100  # Resolution of the grid

# Create the grid
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Checkerboard pattern
length = 4
checkerboard = np.indices((length, length)).sum(axis=0) % 2

# Sample points in regions where checkerboard pattern is 1
sampled_points = []
while len(sampled_points) < N:
    # Randomly sample a point within the x and y range
    x_sample = np.random.uniform(x_min, x_max)
    y_sample = np.random.uniform(y_min, y_max)

    # Determine the closest grid index
    i = int((x_sample - x_min) / (x_max - x_min) * length)
    j = int((y_sample - y_min) / (y_max - y_min) * length)

    # Check if the sampled point is in a region where checkerboard == 1
    if checkerboard[j, i] == 1:
        sampled_points.append((x_sample, y_sample))

# Convert to NumPy array for easier plotting
sampled_points = np.array(sampled_points)

# Plot the checkerboard pattern
plt.figure(figsize=(6, 6))
plt.imshow(checkerboard, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=ListedColormap(["purple", "yellow"]))

# Plot sampled points
plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color="red", marker="o")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.show()
##
# sample time t
# t = 0 p_0 (normal) noise distribution
# t = 1 target (data) distribution
currT = np.random.uniform()
currT = 0.8
# sample data points
N = 100000
# Sample points in regions where checkerboard pattern is 1
sampled_points = []
while len(sampled_points) < N:
    # Randomly sample a point within the x and y range
    x_sample = np.random.uniform(x_min, x_max)
    y_sample = np.random.uniform(y_min, y_max)

    # Determine the closest grid index
    i = int((x_sample - x_min) / (x_max - x_min) * length)
    j = int((y_sample - y_min) / (y_max - y_min) * length)

    # Check if the sampled point is in a region where checkerboard == 1
    if checkerboard[j, i] == 1:
        sampled_points.append((x_sample, y_sample))

# Convert to NumPy array for easier plotting
CurrSamplP = np.array(sampled_points)
data = torch.from_numpy(CurrSamplP).to(dtype=torch.float32)

#x1 = torch.from_numpy(CurrSamplP).to(dtype=torch.float32)


# sample from Noise
NoiseSampl = np.random.multivariate_normal(np.zeros(2),np.eye(2), size = N)
#x0 = torch.from_numpy(NoiseSampl).to(dtype=torch.float32)

#xT = torch.from_numpy((1 -  currT) *  NoiseSampl + currT * CurrSamplP).to(dtype=torch.float32)

# define vector field/flow that satisfies the boundary conditions
# at t = 0 p_0 at t = 1 delta x_1
print('learn Parameters')

##
# Define the input and output dimensions
dim_features = 2  # Number of input features
t_features = 1
channel_features = 512 # Number of output features
# linear_layer1 = nn.Linear(in_features=dim_features, out_features=channel_features )
# linear_layer2 = nn.Linear(in_features=channel_features , out_features=channel_features )
# linear_layer3 = nn.Linear(in_features=channel_features , out_features=dim_features)
#

# Cosine activation
class CosActivation(nn.Module):
    def forward(self, x):
        #return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return torch.cos(x)

# Sine activation
class SinActivation(nn.Module):
    def forward(self, x):
        #return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return torch.sin(x)

# model = nn.Sequential(
#     nn.Linear(in_features=dim_features , out_features=channel_features),
# nn.ReLU(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     nn.ReLU(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     nn.ReLU(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     nn.ReLU(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     nn.ReLU(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     nn.ReLU(),
#     nn.Linear(in_features=channel_features, out_features=dim_features )
# )

model = nn.Sequential(
    nn.Linear(in_features=dim_features + t_features, out_features=channel_features),
    nn.ELU(),
    nn.Linear(in_features=channel_features, out_features=channel_features),
    nn.ELU(),
    nn.Linear(in_features=channel_features, out_features=channel_features),
    nn.ELU(),
    nn.Linear(in_features=channel_features, out_features=channel_features),
    nn.ELU(),
    nn.Linear(in_features=channel_features, out_features=channel_features),
    nn.ELU(),
    nn.Linear(in_features=channel_features, out_features=channel_features),
    nn.ELU(),
    nn.Linear(in_features=channel_features, out_features=dim_features )
)

# model = nn.Sequential(
#     nn.Linear(in_features=dim_features , out_features=channel_features),
#     CosActivation(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     SinActivation(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     CosActivation(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     SinActivation(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     CosActivation(),
#     nn.Linear(in_features=channel_features, out_features=channel_features),
#     SinActivation(),
#     nn.Linear(in_features=channel_features, out_features=dim_features )
# )

import ot
def resample_optimal_plan(source: torch.tensor, target: torch.tensor) -> torch.tensor:
    """
    https://github.com/ulrikisdahl/Conditional-Flow-Matching/blob/main/train.py
    Mini-batch sampling of optimal transport plan between two distributions
    Uses euclidian distance measure for cost and Earth Movers Distance for optimal plan
    """
    #Assume equal mass
    source_weights = torch.ones(source.shape[0]) / source.shape[0]
    target_weights = torch.ones(target.shape[0]) / target.shape[0]

    dist = torch.cdist(source.view(source.shape[0], -1), target.view(source.shape[0], -1))**2
    plan = ot.emd(source_weights, target_weights, dist)
    pairs = torch.argwhere(plan > 0)

    #reorder (along the batch dimension only) the target distribution according to corresponding source pairing
    target = target[pairs[:, 1]]
    return source, target

learnModel = False
if learnModel:

    optim = torch.optim.AdamW( model.parameters(), lr=1e-4)
    batch_size = 64
    k = 0
    loss = torch.tensor(2)
    prevloss = torch.tensor(2.5)
    #while  abs(prevloss-loss) > 1e-6:
    while k < 1e5:
        k += 1
        prevloss = loss.detach().clone()

        # draw samples from target
        x1 = data[torch.randint(data.size(0), (batch_size,))]

        # draw time t
        t = torch.rand(x1.size(0))
        # t = torch.ones(x1.size(0))

        # draw samples from noise
        NoiseSampl = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=batch_size)
        x0 = torch.from_numpy(NoiseSampl).to(dtype=torch.float32)
        x0, x1 = resample_optimal_plan(x0, x1)
        # xt
        sample_input = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred = model(torch.cat([sample_input, torch.unsqueeze(t, 1)], dim=1) )
        target = x1 - x0
        #target = x1

        loss = ((target - pred)**2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        #print(loss)

    # learn parameters
    print(f'rounds {k}')
    print(loss)
    # Save the model's state_dict
    PATH = "SimpleModel.pth"  # Recommended file extension is .pt or .pth
    torch.save(model.state_dict(), PATH)
else:
    # Load the saved state_dict
    PATH = "SimpleModel.pth"
    model.load_state_dict(torch.load(PATH))
    model.eval()



##
test_size = 500
plot_every = 50
plotBetween = True
x1 = data[torch.randint(data.size(0), (test_size,))]

#x0 = torch.from_numpy(NoiseSampl).to(dtype=torch.float32)
xt = torch.randn(test_size, 2)
#x0, xt = resample_optimal_plan(x0, xt)
steps = 500
for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
#for t in torch.linspace(0, 1, steps):
    pred = model(torch.cat([xt, torch.unsqueeze(torch.ones(xt.size(0)) * t , 1)], dim=1))
    #pred = model(xt)
    xt = xt + (1 / steps) * pred
    if i % plot_every == 0 and plotBetween:
        plt.figure(figsize=(6, 6))
        plt.scatter(x1[:, 0].detach().numpy(), x1[:, 1].detach().numpy(), color="red", marker="o")
        plt.scatter(xt[:, 0].detach().numpy(), xt[:, 1].detach().numpy(), color="green", marker="o")
        #plt.show(block = True)



# t = 1
# xt = torch.randn(test_size, 2)
# pred = model(torch.cat([xt, torch.unsqueeze(torch.ones(xt.size(0)) * t, 1)], dim=1))
# xt = xt + pred


# Plot the checkerboard pattern
plt.figure(figsize=(6, 6))
#plt.imshow(checkerboard, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=ListedColormap(["purple", "yellow"]))
#pred = xt + model(xt)
#pred = linear_layer(sample_input);
# Plot sampled points
plt.scatter(x1[:, 0].detach().numpy() , x1[:, 1].detach().numpy() , color="red", marker="o", label = 'target')
plt.scatter(xt[:, 0].detach().numpy() , xt[:, 1].detach().numpy() , color="black", marker="o", label = 'prediction')
plt.xlabel("X-axis")
plt.legend()
plt.ylabel("Y-axis")
plt.show(block = True)
print(f'done')