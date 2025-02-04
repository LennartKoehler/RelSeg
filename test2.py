import torch
import matplotlib.pyplot as plt
import torch.nn as nn

result = torch.load("test_result.pkl")
read = torch.load("test_read.pkl")
print(read)
print(result[0])
result = result[1]
plt.plot(read.signal)
for segment in result["segments"]:
    plt.vlines(segment.detach().cpu(), -2, 2)
plt.xlim(0,500)
plt.savefig("test.png")