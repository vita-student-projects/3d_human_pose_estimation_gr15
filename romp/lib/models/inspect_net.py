import resnet_50, efficient_net, hrnet_32, romp_model
import torch
from torchsummary import summary
from torchviz import make_dot
import numpy as np
import sys

import torchvision.models as models

# net = resnet_50.ResNet_50().cpu()
# summary(net, (512,512,3), device="cpu")
# dot = make_dot(net(torch.rand(size=(1,512,512,3)).cpu()), params=dict(net.named_parameters()))

# dot.render(filename='resnet_graph', format='pdf')


# input("This is was the ResNet\nPress [Enter] to continue ...")

net = hrnet_32.HigherResolutionNet().cpu()
summary(net, (512,512,3,), device="cpu")
pytorch_total_params = sum(p.numel() for p in net.parameters())

print("Params:", pytorch_total_params)
input("This is was the HRNet\nPress [Enter] to continue ...")


net = efficient_net.EfficientNetRomp().cpu()
summary(net, (512,512,3), device="cpu")
dot = make_dot(net(torch.rand(size=(1,512,512,3,)).cpu()), params=dict(net.named_parameters()))

dot.render(filename='effnet_graph', format='pdf')

input("This is was the HRNet\nPress [Enter] to continue ...")

net = romp_model.ROMP(net).cpu()
pytorch_total_params = sum(p.numel() for p in net.parameters())

print("ROMP (COMPLETE!)")
print("Params:", pytorch_total_params)
# print("Sz:", sys.getsizeof(net))



input("This is was the Head\nPress [Enter] to continue ...")


# Use the torchsummary library to get a summary of your network



# # Create a visualization of your network using torchviz
# dot = make_dot(net(torch.rand(size=(1,3,512,512,)).cuda()), params=dict(net.named_parameters()))

# dot.render(filename='net_graph', format='pdf')

# # Print out the number of parameters in your network
# num_params = sum(p.numel() for p in net.parameters())
# print(f"Number of parameters in network: {num_params}")