import torch
import torch.nn as nn
from nmp.model.pointnet import PointNet
from rlkit.torch import pytorch_util as ptu


class Forward(nn.Module):
    def __init__(self,
                 feature_dim,
                 action_dim,
                 #hidden_size=256,
                 #hidden_init=ptu.fanin_init,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        #self.hidden_init = hidden_init

        self.fc = nn.Sequential(nn.Linear(feature_dim+action_dim, 256),
                                nn.ELU(),
                                nn.Linear(256, feature_dim))
        # self.fc = nn.Linear(feature_dim+action_dim, feature_dim)
        #self.hidden_init(self.fc.weight, scale=1.0)

    def forward(self, features, action):
        x = torch.cat((features, action), 1)
        return self.fc(x)


class Inverse(nn.Module):
    def __init__(self,
                 feature_dim,
                 action_dim,
                 #hidden_init=ptu.fanin_init,
                 ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        #self.hidden_init = hidden_init

        #self.fc = nn.Linear(2*feature_dim, action_dim)
        self.fc = nn.Sequential(nn.Linear(2 * feature_dim, 256),
                                nn.ELU(),
                                nn.Linear(256, action_dim))
        #self.hidden_init(self.fc.weight, scale=1.0)

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
        super().__init__(hidden_sizes=hidden_sizes, **kwargs)
        self.forward_model = Forward(feature_dim=hidden_sizes[-1],
                                     action_dim=action_dim)

        self.inverse_model = Inverse(feature_dim=hidden_sizes[-1],
                                     action_dim=action_dim)


    def forward(self, action, obs, next_obs):
        h = super().forward(obs, return_features=True)
        next_h = super().forward(next_obs, return_features=True)

        # forward_input = torch.cat((h, action), 1)
        # inverse_input = torch.cat((h, next_h), 1)

        next_h_pred = self.forward_model(h, action)
        action_pred = self.inverse_model(h, next_h)

        return next_h, next_h_pred, action_pred






