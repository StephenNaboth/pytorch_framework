# Import dependenies
import torch
import torch.nn as nn  # artificial neural network

# 1) Design Model, the model has to implement the forward pass!
# Here we could simply use a built-in model from PyTorch
# model = nn.Linear(input_size, output_size)
# Linear regression
# f = w * x
# here : f = 2 * x

# 0) Training samples, watch the shape!
X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12],
                 [14], [16]], dtype=torch.float32)


n_samples, n_features = X.shape

# 0) create a test sample

X_test = torch.tensor([5], dtype=torch.float32)


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        # has been simplified in python3
        super(LinearRegression, self).__init__()
        # we define different layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


input_size, output_size = n_features, n_features


model = LinearRegression(input_size, output_size)

print(
    f'Prediction before training: f({X_test.item()}) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01

n_epochs = 300

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(n_epochs):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if (epoch+1) % 5 == 0:
        w, b = model.parameters()  # unpack parameters w -weights and b - bias term
        print('epoch ', epoch+1, ': w = ',
              w[0][0].item(), ' loss = ', l.item())

print(
    f'Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}')
