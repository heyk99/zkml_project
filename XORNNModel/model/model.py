import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])


class TinyMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x


model = TinyMLP()

# 关键修正 1️⃣
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01     # ← 改小
)

loss_fn = nn.MSELoss()

# 关键修正 2️⃣
for epoch in range(5000):   # ← 多训练一点

    pred = model(X)

    loss = loss_fn(pred, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 500 == 0:
        print(epoch, loss.item())


# Test
print("\nTest:")

with torch.no_grad():

    for i in range(4):

        out = model(X[i])

        print(X[i], out)


# Export ONNX
dummy_input = torch.randn(1, 2)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("model.onnx generated")