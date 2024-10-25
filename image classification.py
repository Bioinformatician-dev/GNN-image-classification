import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torchvision import datasets, transforms
import numpy as np

# Parameters
num_classes = 10  # For example, 10 classes for CIFAR-10
num_nodes = 32 * 32  # Assuming images are 32x32 pixels
num_channels = 3  # RGB images

# Load and preprocess the dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Function to create a graph from an image
def create_graph(image):
    image = image.view(-1, num_channels)  # Flatten the image
    edge_index = torch.tensor(np.array(np.nonzero(np.eye(num_nodes))), dtype=torch.long)  # Simplistic self-connections
    return Data(x=image, edge_index=edge_index)

# Initialize model
model = GNN(input_dim=num_channels, hidden_dim=16, output_dim=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(10):  # Adjust epochs as needed
    for images, labels in train_loader:
        graph = create_graph(images[0])
        optimizer.zero_grad()
        out = model(graph)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'gnn_image_classifier.pth')
