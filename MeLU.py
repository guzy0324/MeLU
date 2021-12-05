import torch
import numpy as np
from copy import deepcopy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

from embeddings import item, user


class Linear(torch.nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x, a=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear, self).forward(x)
        return out


class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.item_emb = item(config)
        self.user_emb = user(config)
        self.fc1 = Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = Linear(self.fc2_out_dim, 1)
        self.final_part = torch.nn.Sequential(self.fc1, torch.nn.ReLU(), self.fc2, torch.nn.ReLU(), self.linear_out)

    def forward(self, x, training = True):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        x = self.final_part(x)
        return x


class MeLU(torch.nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.local_lr = config['local_lr']
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        fast_parameters = self.model.final_part.parameters()
        for idx in range(num_local_update):
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            # local update
            for j, weight in enumerate(self.model.final_part.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.local_lr * grad[j]
                else:
                    weight.fast = weight.fast - self.local_lr * grad[j]
                fast_parameters.append(weight.fast)
        query_set_y_pred = self.model(query_set_x)
        for weight in self.model.final_part.parameters():
            weight.fast = None
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()
        self.meta_optim.step()
        return

    def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
        tmp = 0.
        if self.cuda():
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()
        fast_parameters = self.model.final_part.parameters()
        for idx in range(num_local_update):
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # unit loss
            loss /= torch.norm(loss).tolist()
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            for j, weight in enumerate(self.model.final_part.parameters()):
                # For averaging Forbenius norm.
                tmp += torch.norm(grad[j])
                if weight.fast is None:
                    weight.fast = weight - self.local_lr * grad[j]
                else:
                    weight.fast = weight.fast - self.local_lr * grad[j]
                fast_parameters.append(weight.fast)
        for weight in self.model.final_part.parameters():
            weight.fast = None
        return tmp / num_local_update

