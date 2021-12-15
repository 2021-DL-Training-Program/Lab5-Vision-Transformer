import torch
import torch.nn as nn

import numpy as np
import math
from transformers import ViTModel
from dataloader import get_loader
from torchvision import transforms
import pickle
from build_vocab import Vocabulary
from tqdm import tqdm
import argparse

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hid_dim=768
        self.proj = nn.Linear(768, 768)
        
    def forward(self, src):
        #return = [batch size, patch len, hid dim]
        return self.proj(self.vit(pixel_values=src).last_hidden_state)

model = Encoder()