import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
device = torch.device("cuda:0")

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ODEFunc(nn.Module):
    def __init__(self, input_size=5, output_size=2, hidden_size=256):
        super(ODEFunc, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        """
        Perform one step in solving ODE.
        """
        # print ("shape****************", x.shape)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x


class DiffeqSolver(nn.Module):

    def __init__(self, ode_func, method, odeint_rtol, odeint_atol):
        super(DiffeqSolver, self).__init__()
        self.ode_func = ode_func
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.method = method

    # TODO: fix this
    def forward(self, time_steps, hxs, actions, masks):
        """
            Decode the trajectory through ODE Solver
            @:param first_point, shape [N, D]
                    time_steps, shape [N, T,]
            @:return predicted the trajectory, shape [N, D]
        """

        # if done, reset hidden_state to zero
        # print ("**************", hxs.shape, masks.shape)
        try:
            hxs = (hxs * masks).unsqueeze(0)
        except:
            # print ("initializing")
            hxs = torch.zeros((masks.shape[0], 64)).to(device)

        #dummy, # why not 2d?
        # time_steps = torch.FloatTensor([0, 1]).to(device)

        # print ("**************", hxs.squeeze(0).shape, actions.shape)
        # first_point = torch.cat((hxs.squeeze(0), actions), axis=1)

        first_point = hxs.squeeze(0)
        #
        # pred = odeint(self.ode_func, first_point, time_steps,
        #               rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.method)  # [T, N, D]
        #
        pred = []
        for i in range(len(time_steps)):
            if time_steps[i][0] == 0:
                pred.append(first_point[i])
            else:
                pred.append(odeint(self.ode_func, first_point[i], time_steps[i],
                            rtol=self.odeint_rtol, atol=self.odeint_atol, method="explicit_adams")[-1])  # [T, N, D]

        pred = torch.stack(pred,dim=0)

        # assert (torch.mean(pred[0, :, :] - first_point) < 0.001)  # the first prediction is same with first point
        return pred

        # pred = pred.permute(1, 0, 2)  # [N, T, D]
        # assert (torch.mean(pred[:, 0, :] - first_point) < 0.001)  # the first prediction is same with first point
        # assert pred.size(0) == first_point.size(0)
        # assert pred.size(1) == time_steps.size(0)
        # assert pred.size(2) == first_point.size(1)
        # return pred


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks, actions=None):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, actions=None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, actions=actions)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, actions=None):
        value, _, _ = self.base(inputs, rnn_hxs, masks, actions=actions)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, actions=None):

        # print("************", inputs.shape, rnn_hxs.shape, masks.shape)
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, actions=actions)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, use_ode=False):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self.use_ode = use_ode

        if use_ode:
            ode_tol = 1e-8

            # print ("ODEFUNC: ****************", recurrent_input_size, recurrent_input_size)

            # self.gen_ode_func = ODEFunc(input_size=recurrent_input_size+3, output_size=recurrent_input_size)
            # self.gen_ode_func = ODEFunc(input_size=64+3, output_size=64)
            self.gen_ode_func = ODEFunc(input_size=64 , output_size=64)
            self.diffeq_solver = DiffeqSolver(self.gen_ode_func, 'dopri5', ode_tol, ode_tol / 10)

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):

        if self._recurrent:
            return self._hidden_size
        print ("recurrent_hidden_size is one *******************")
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks, actions=None):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, use_ode=False):
        super(MLPBase, self).__init__(recurrent, num_inputs-2, hidden_size, use_ode=use_ode)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    # TODO: get timesteps, actions
    def forward(self, observations, rnn_hxs, masks, actions=None):
        time_steps = observations[:, -2:]
        x = observations[:, :-2]
        
        if self.use_ode:
            # time_steps, hxs, action, masks,
            # dummy time-steps for now
            # print ("rnn_hxs", rnn_hxs.shape, actions.shape, masks.shape)
            ## rnn_hxs 8x64, actions 8x3, masks 8x1
            latent_state = self.diffeq_solver(time_steps, rnn_hxs, actions, masks)
            x, rnn_hxs = self._forward_gru(x, latent_state, masks)

        elif self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)


        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
