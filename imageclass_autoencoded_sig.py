import torch
import torch.nn as nn
import torch.optim as optim

# Defining the structure of the NN
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # Input layer size 16 to hidden layer size 8
        self.fc1 = nn.Linear(16, 8)
        # Hidden layer size 8 to hidden layer size 4
        self.fc2 = nn.Linear(8, 4)

        # Hidden layer size 4 to latent layer size 8
        self.fc3 = nn.Linear(4, 8)
        # Latent layer size 8 to output layer size 16
        self.fc4 = nn.Linear(8, 16)

        # Activation functions
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder: shrink dimensions
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        # Latent space representation
        latent = self.sigmoid(self.fc3(x))

        # Decoder: reconstruct back to input size
        x = self.sigmoid(self.fc4(latent))

        return x
    
    def forward_latent(self, x):
        # Encoder forward pass to latent space
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x

# Instantiate the model
model = SimpleNN()
cost = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Input data
input_data = torch.tensor([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sample 1
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Sample 2
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # Sample 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Sample 4
], dtype=torch.float32)

# Initial Model Output Before Training
initial_output = model(input_data)

# Display Weights and Biases before training
print("----------------------Before Training----------------------")
print("\nfc1 Weights:", model.fc1.weight)
print("fc1 Biases:", model.fc1.bias)
print("\nfc2 Weights:", model.fc2.weight)
print("fc2 Biases:", model.fc2.bias)

# Display Output before training
print("\nInitial Model Output:")
print(initial_output)

# Initial loss
initial_loss = cost(initial_output, input_data)
print(f"\nInitial Loss: {initial_loss.item()}")

# Training loop
for i in range(10000):
    optimizer.zero_grad()  # Zero the gradients
    output = model(input_data)  # Forward pass
    loss = cost(output, input_data)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

# Model Output After Training
final_output = model(input_data)

# Display Weights and Biases after training
print("\n----------------------After Training----------------------")
print("\nfc1 Weights:", model.fc1.weight)
print("fc1 Biases:", model.fc1.bias)
print("\nfc2 Weights:", model.fc2.weight)
print("fc2 Biases:", model.fc2.bias)

# Display Output after training
print("\nFinal Model Output:")
print(final_output)

# Final loss
final_loss = cost(final_output, input_data)
print(f"\nFinal Loss: {final_loss.item()}")

# Test Input 
test_input = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
test_output = model.forward_latent(test_input)
print(f"\nTest Output from Latent Space Representation: {test_output}")
