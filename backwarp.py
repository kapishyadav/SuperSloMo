import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class backwarp(nn.Module):
    #Use a flow to warp one image to another.
	#Implement the I = g(I, F) function
    #https://nanonets.com/blog/optical-flow/
    #http://www.jarnoralli.fi/joomla/research/topics/optical-flow

    def __init__(self, width, height, device):
        super(backwarp, self).__init__()
        grid        = np.meshgrid(np.arange(width), np.arange(height))
        self.gridX  = torch.tensor(grid[0], requires_grad=False).to(device)
        self.gridY  = torch.tensor(grid[1], requires_grad=False).to(device)
        self.width  = width
        self.height = height

    def forward(self, image, flow):
        x, y = flow[:, 0, :, :], flow[:, 1, :, :]        
        x = self.normalize(self.gridX.unsqueeze(0).expand_as(x).float() + x, "W")
        y = self.normalize(self.gridY.unsqueeze(0).expand_as(y).float() + y, "H")
        warpedImg = torch.nn.functional.grid_sample(image, torch.stack((x,y), dim=3), align_corners=True)
        return warpedImg

    def normalize(self, flow, dim):
        if dim == "W":
            return 2*(flow/self.width - 0.5)
        else:
            return 2*(flow/self.height - 0.5)
