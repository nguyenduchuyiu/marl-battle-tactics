import torch.nn as nn

class SunTzuNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(SunTzuNetwork, self).__init__()
        
        height, width, channels = observation_shape
        
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, observation_shape[-1], kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(observation_shape[-1]),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], kernel_size=3, stride=1, padding=1),     
            nn.ReLU(),
            nn.BatchNorm2d(observation_shape[-1]),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], kernel_size=3, stride=1, padding=1),    
            nn.ReLU(),
        )
        
        # Calculate flattened size after CNN
        flattened_size = observation_shape[-1] * height * width
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change from (1, 13, 13, 5) to (1, 5, 13, 13)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1) 
        x = self.fc(x)
        return x
