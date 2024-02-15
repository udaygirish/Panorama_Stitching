import torch
import sys
import torch.nn as nn
import os
import sys

sys.path.append("../")
from Network.Network import Supervised_Net, Unsupervised_Net
from torchview import draw_graph


# Add path

# sys.path.append("./")

# Don't generate pyc codes
sys.dont_write_bytecode = True

model_graph = draw_graph(
    Supervised_Net((1, 2, 128, 128), 8),
    input_size=(1, 2, 128, 128),
    graph_name="Supervised_Net",
    roll=True,
    save_graph=True,
)

# model_graph.visual_graph()
