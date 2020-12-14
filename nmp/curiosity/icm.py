import torch
import torch.nn as nn
from nmp.model.pointnet import PointNet


class Forward(nn.Module):
    def __init__(self,
                 feature_dim,
                 action_dim,
                 hidden_size = 256,):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.mlp = nn.Sequential(nn.Linear(feature_dim+action_dim, hidden_size),
                                 nn.ELU(),
                                 nn.Linear(hidden_size, feature_dim))

    def forward(self, features, action):
        x = torch.cat((features, action),1)
        return self.mlp(x)


class Inverse(nn.Module):
    def __init__(self,
                 feature_dim,
                 action_dim,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim

        self.fc = nn.Linear(2*feature_dim, action_dim)

    def forward(self, features, next_features):
        x = torch.cat((features, next_features), 1)
        predicted_action = torch.tanh(self.fc(x))
        return predicted_action


class ICM(PointNet):
    def __init__(
            self,
            hidden_sizes,
            action_dim,
            **kwargs
    ):
        super().__init__(hidden_sizes=hidden_sizes, output_size=1, **kwargs)
        self.forward_model = Forward(feature_dim=hidden_sizes[-1],
                                     action_dim=action_dim)

        self.inverse_model = Inverse(feature_dim=hidden_sizes[-1],
                                     action_dim=action_dim)



    def forward(self, action, obs, next_obs):
        h = super().forward(obs, return_features=True)
        next_h = super().forward(next_obs, return_features=True)

        forward_input = torch.cat((h, action))
        inverse_input = torch.cat((h, next_h))

        next_h_pred = self.forward_model(forward_input)
        action_pred = self.inverse_model(inverse_input)

        return next_h, next_h_pred, action_pred






