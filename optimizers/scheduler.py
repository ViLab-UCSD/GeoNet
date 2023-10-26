from collections import Counter

class MultiStep_scheduler(object):

    def __init__(self, init_lr, optimizer, milestones, gamma):
        super(MultiStep_scheduler, self).__init__()
        self.opt = optimizer
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.lr = init_lr
        self.iteration = 0
        self.group_ratios = [p["lr"] for p in self.opt.param_groups]

        for i,param_group in enumerate(self.opt.param_groups):
                param_group['lr'] = self.lr*self.group_ratios[i]

    def step(self):
        if self.iteration in self.milestones:
            for param_group in self.opt.param_groups:
                param_group['lr'] = param_group['lr'] * self.gamma**self.milestones[self.iteration]
        self.iteration += 1
            

    def get_lr(self):
        return self.opt.param_groups[0]['lr']

    def state_dict(self):
        return {
            'lr': self.get_lr(),
            'iter': self.iteration,
            'milestones': self.milestones,
            'decay_rate': self.gamma
        }

    def load_state_dict(self, state_dict):
        self.iteration = state_dict['iter']
        self.milestones = state_dict['milestones']
        self.gamma = state_dict['gamma']
        self.step()

class step_scheduler(object):

    def __init__(self, init_lr, optimizer, step_size, gamma):
        super(step_scheduler, self).__init__()
        self.opt = optimizer
        self.step_size = step_size
        self.decay_rate = gamma
        self.lr = init_lr
        self.epoch = 1

    def step(self):
        if self.epoch % self.step_size == 0:
            for param_group in self.opt.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay_rate
        self.epoch += 1

    def get_lr(self):
        return self.opt.param_groups[0]['lr']

    def state_dict(self):
        return {'lr': self.get_lr(),
            'epoch': self.epoch,
            'step_size': self.step_size,
            'decay_rate': self.decay_rate
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.step_size = state_dict['step_size']
        self.decay_rate = state_dict['decay_rate']
        self.step()


class inv_scheduler(object):

    def __init__(self, init_lr, optimizer, gamma, power):
        super(inv_scheduler, self).__init__()
        self.opt = optimizer
        self.gamma = gamma
        self.power = power
        self.iteration = 0
        self.lr = init_lr
        self.weight_decay = self.opt.param_groups[0]['weight_decay']
        self.group_ratios = [p["lr"] for p in self.opt.param_groups]

    def step(self):
        lr = self.lr * (1 + self.gamma * self.iteration) ** (-self.power)
        for ii, param_group in enumerate(self.opt.param_groups):
            param_group['lr'] = lr * self.group_ratios[ii]
            # param_group['weight_decay'] = self.weight_decay * param_group['decay_mult']
        self.iteration += 1

    def state_dict(self):
        return {'lr': self.lr,
            'weight_decay': self.weight_decay,
            'iteration': self.iteration,
            'gamma': self.gamma,
            'power': self.power
        }

    def load_state_dict(self, state_dict):
        self.lr = state_dict['lr']
        self.weight_decay = state_dict['weight_decay']
        self.iteration = state_dict['iteration']
        self.gamma = state_dict['gamma']
        self.power = state_dict['power']
        self.step()


