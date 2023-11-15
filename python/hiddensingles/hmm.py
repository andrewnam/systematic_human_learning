import einops
from functools import cached_property
import torch
from torch import nn
from .misc import TensorDict, torch_utils as tu


class HiddenMarkovModel(nn.Module):

    def __init__(self, subject_responses):
        super().__init__()
        self.subject_responses = nn.Parameter(subject_responses, requires_grad=False)  # [num_subjects, num_trials]

        self.w_strategy_success = nn.Parameter(torch.rand(4, 4))
        mask = torch.full((4, 4), -torch.inf).triu(diagonal=1)
        self.strategy_success_mask = nn.Parameter(mask, requires_grad=False)

        self.p_strategy_responses = nn.Parameter(torch.tensor([
            [3 / 9, 4 / 9, 1 / 9, 1 / 9],
            [0, 4 / 6, 1 / 6, 1 / 6],
            [0, 0, 1 / 2, 1 / 2],
            [0, 0, 0, 1]
        ]), requires_grad=False)

        self.w_state_priors = nn.Parameter(torch.rand(4))
        self.w_transitions = nn.Parameter(torch.rand(4, 4))
        mask = torch.full((4, 4), -torch.inf).tril(diagonal=-1)
        self.transition_mask = nn.Parameter(mask, requires_grad=False)

    @property
    def num_subjects(self):
        return self.subject_responses.shape[0]

    @property
    def num_trials(self):
        return self.subject_responses.shape[1]

    @cached_property
    def params(self):
        with torch.no_grad():
            return TensorDict(p_state_priors=self.get_p_state_priors(),
                              p_transitions=self.get_p_transitions(),
                              p_strategy_success=self.get_p_strategy_success(),
                              p_state_responses=self.get_p_state_responses())

    @cached_property
    def paths(self):
        """
        Returns a tensor containing every possible path
        return: tensor with shape [num_paths, num_trials]
        """
        paths = [[0], [1], [2], [3]]
        for _ in range(self.num_trials - 1):
            new_paths = []
            for path in paths:
                for s in range(path[-1], 4):
                    new_paths.append(path + [s])
            paths = new_paths
        return torch.tensor(paths, device=self.subject_responses.device)

    def get_p_strategy_success(self):
        """
        Compute P(response | strategy)
        """
        return (self.w_strategy_success + self.strategy_success_mask).softmax(-1)

    def get_p_state_responses(self):
        """
        Compute P(response | state)
        """
        return self.get_p_strategy_success().mm(self.p_strategy_responses)

    def get_p_state_priors(self):
        """
        Compute P(state_0)
        """
        return self.w_state_priors.softmax(-1)

    def get_p_transitions(self):
        """
        Compute P(state_{n+1} | state_{n})
        """
        return (self.w_transitions + self.transition_mask).softmax(-1)

    def get_p_states(self):
        """
        Compute P(state | trial)
        """
        p_transitions = self.get_p_transitions()

        p_s = self.get_p_state_priors().unsqueeze(0)
        p_states = [p_s]
        for i in range(self.num_trials - 1):
            p_s = p_s.mm(p_transitions)
            p_states.append(p_s)
        p_states = torch.cat(p_states, dim=0)
        return p_states

    def get_p_responses(self):
        """
        Compute P(response | trial), i.e. the aggregate response distribution at each trial
        returns: tensor with shape [num_trials, 4]
        """
        return self.get_p_states().mm(self.get_p_state_responses())

    def get_p_paths(self):
        """
        Compute P(path) using just p_state_priors and p_transitions.
        returns: tensor with shape [num_paths]
        """
        prev_states = self.paths[:, :-1]
        next_states = self.paths[:, 1:]
        p_path_transitions = self.get_p_transitions()[prev_states]
        p_path_transitions = tu.select_subtensors(p_path_transitions, next_states)
        p_path0 = self.get_p_state_priors()[self.paths[:, 0]]
        p_paths = torch.cat([p_path0.unsqueeze(-1), p_path_transitions], dim=-1)
        p_paths = p_paths.log().sum(-1).exp()  # for numerical stability
        return p_paths

    def get_response_likelihoods(self):
        """
        Computes P(responses | path) for each subject.
        return: tensor with shape [num_subjects, num_paths]
        """
        p_responses = self.get_p_state_responses()[self.paths]
        p_responses = einops.repeat(p_responses, "paths trials r -> subj paths trials r", subj=self.num_subjects)
        responses = einops.repeat(self.subject_responses, "subj trials -> subj paths trials", paths=len(self.paths))
        trial_likelihoods = tu.select_subtensors(p_responses, responses)  # [num_subjects, num_paths, num_trials]
        path_likelihoods = trial_likelihoods.log().sum(-1).exp()  # [num_subjects, num_paths]
        return path_likelihoods

    def get_path_posteriors(self):
        """
        Computes P(path | responses) for each subject.
        return: tensor with shape [num_subjects, num_paths]
        """
        priors = self.get_p_paths()
        likelihoods = self.get_response_likelihoods()
        numerator = priors.unsqueeze(0) * likelihoods
        denominator = numerator.sum(-1)
        posteriors = numerator / denominator.unsqueeze(-1)
        return posteriors

    def get_loss(self):
        """
        Computes NLL between p_responses and subject_responses
        """
        if 'params' in self.__dict__:
            del self.params  # presumably, the loss will be used to update params

        p_responses = self.get_p_responses()
        p_responses = einops.repeat(p_responses, "t f -> n t f", n=self.num_subjects)
        loss = tu.nll_loss(p_responses.log(), self.subject_responses)
        return loss
