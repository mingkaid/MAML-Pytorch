import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy
import itertools



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.log_noise_init_q = 2 * np.log(0.15)
        self.log_noise_init_p = self.log_noise_init_q
        self.log_gamma_init = 2 * np.log(self.update_lr)


        self.net = Learner(config, args.imgc, args.imgsz)
        
        # this dict contains all tensors needed to be optimized
        self.log_gamma_q = nn.ParameterList()
        self.log_gamma_p = nn.ParameterList()
        self.log_v_q = nn.ParameterList()
        self.log_sigma2 = nn.ParameterList()

        for i, (name, param) in enumerate(config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                g_q = nn.Parameter(torch.ones(*param[:4]) * self.log_gamma_init)
                # torch.nn.init.constant_(g_q, self.update_lr)
                self.log_gamma_q.append(g_q)
                # [ch_out]
                self.log_gamma_q.append(nn.Parameter(torch.ones(param[0]) * self.log_gamma_init))
                
                # [ch_out, ch_in, kernelsz, kernelsz]
                g_p = nn.Parameter(torch.ones(*param[:4]) * self.update_lr)
                # torch.nn.init.constant_(g_p, self.update_lr)
                self.log_gamma_p.append(g_p)
                # [ch_out]
                self.log_gamma_p.append(nn.Parameter(torch.ones(param[0]) * self.log_gamma_init))
                
                # [ch_out, ch_in, kernelsz, kernelsz]
                v = nn.Parameter(torch.ones(*param[:4]) * self.log_noise_init_q)
                # torch.nn.init.constant_(v, 1)
                self.log_v_q.append(v)
                # [ch_out]
                self.log_v_q.append(nn.Parameter(torch.ones(param[0]) * self.log_noise_init_q))
                
                # [ch_out, ch_in, kernelsz, kernelsz]
                s = nn.Parameter(torch.ones(*param[:4]) * self.log_noise_init_p)
                # torch.nn.init.constant_(s, 1)
                self.log_sigma2.append(s)
                # [ch_out]
                self.log_sigma2.append(nn.Parameter(torch.ones(param[0]) * self.log_noise_init_p))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                g_q = nn.Parameter(torch.ones(*param[:4]) * self.update_lr)
                # torch.nn.init.constant_(g_q, self.update_lr)
                self.log_gamma_q.append(g_q)
                # [ch_in, ch_out]
                self.log_gamma_q.append(nn.Parameter(torch.ones(param[1]) * self.log_gamma_init))
                
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                g_p = nn.Parameter(torch.ones(*param[:4]) * self.update_lr)
                # torch.nn.init.constant_(g_p, self.update_lr)
                self.log_gamma_p.append(g_p)
                # [ch_out]
                self.log_gamma_p.append(nn.Parameter(torch.ones(param[1]) * self.log_gamma_init))
                
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                v = nn.Parameter(torch.ones(*param[:4]) * self.log_noise_init_q)
                # torch.nn.init.constant_(v, 1)
                self.log_v_q.append(v)
                # [ch_out]
                self.log_v_q.append(nn.Parameter(torch.ones(param[1]) * self.log_noise_init_q))
                
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                s = nn.Parameter(torch.ones(*param[:4]) * self.log_noise_init_p)
                # torch.nn.init.constant_(s, 1)
                self.log_sigma2.append(s)
                # [ch_out]
                self.log_sigma2.append(nn.Parameter(torch.ones(param[1]) * self.log_noise_init_p))

            elif name is 'linear':
                # [ch_out, ch_in]
                g_q = nn.Parameter(torch.ones(*param) * self.update_lr)
                # torch.nn.init.constant_(g_q, self.update_lr)
                self.log_gamma_q.append(g_q)
                # [ch_in, ch_out]
                self.log_gamma_q.append(nn.Parameter(torch.ones(param[0]) * self.log_gamma_init))
                
                # [ch_out, ch_in]
                g_p = nn.Parameter(torch.ones(*param) * self.update_lr)
                # torch.nn.init.constant_(g_p, self.update_lr)
                self.log_gamma_p.append(g_p)
                # [ch_in, ch_out]
                self.log_gamma_p.append(nn.Parameter(torch.ones(param[0]) * self.log_gamma_init))
                
                # [ch_out, ch_in]
                v = nn.Parameter(torch.ones(*param) * self.log_noise_init_q)
                # torch.nn.init.constant_(v, 1)
                self.log_v_q.append(v)
                # [ch_in, ch_out]
                self.log_v_q.append(nn.Parameter(torch.ones(param[0]) * self.log_noise_init_q))
                
                # [ch_out, ch_in]
                s = nn.Parameter(torch.ones(*param) * self.log_noise_init_p)
                # torch.nn.init.constant_(s, 1)
                self.log_sigma2.append(s)
                # [ch_in, ch_out]
                self.log_sigma2.append(nn.Parameter(torch.ones(param[0]) * self.log_noise_init_p))

            elif name is 'bn':
                # [ch_out]
                g_q = nn.Parameter(torch.ones(param[0]) * self.update_lr)
                # torch.nn.init.constant_(g_q, self.update_lr)
                self.log_gamma_q.append(g_q)
                # [ch_out]
                self.log_gamma_q.append(nn.Parameter(torch.ones(param[0]) * self.log_gamma_init))
                
                # [ch_out]
                g_p = nn.Parameter(torch.ones(param[0]) * self.update_lr)
                # torch.nn.init.constant_(g_p, self.update_lr)
                self.log_gamma_p.append(g_p)
                # [ch_out]
                self.log_gamma_p.append(nn.Parameter(torch.ones(param[0]) * self.log_gamma_init))
                
                # [ch_out]
                v = nn.Parameter(torch.ones(param[0]) * self.log_noise_init_q)
                # torch.nn.init.constant_(v, 1)
                self.log_v_q.append(v)
                # [ch_out]
                self.log_v_q.append(nn.Parameter(torch.ones(param[0]) * self.log_noise_init_q))
                
                # [ch_out]
                s = nn.Parameter(torch.ones(param[0]) * self.log_noise_init_p)
                # torch.nn.init.constant_(s, 1)
                self.log_sigma2.append(s)
                # [ch_out]
                self.log_sigma2.append(nn.Parameter(torch.ones(param[0]) * self.log_noise_init_p))


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError
        
        params = [self.net.parameters(), self.log_sigma2, self.log_v_q, self.log_gamma_p, self.log_gamma_q]
        self.meta_optim = optim.Adam(itertools.chain(*params), lr=self.meta_lr)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        kls = 0 # KL divergence between variational distribution and data distribution
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):
            
            # 1. Evaluate the test loss of mu_theta
            logits = self.net(x_qry[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_qry[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            # 1.5 Sample theta from the updated variational distribution
            mu_theta_test = list(map(lambda p: p[0] - torch.exp(0.5 * p[1]) * p[2],
                                     zip(self.net.parameters(), self.log_gamma_q, grad)))
            # theta = mu_theta_test
            theta = list(map(lambda p: p[0] + torch.exp(0.5 * p[1]) * torch.randn(p[1].shape).type_as(p[1]),
                             zip(mu_theta_test, self.log_v_q)))
            
            # 2. run the i-th task with theta and compute loss for k=0
            logits = self.net(x_spt[i], vars=theta, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, theta)
            # 2.5 Start computing adapted parameters with gradient descent:
            phi_i = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, theta)))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], theta, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], phi_i, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], phi_i, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, phi_i)
                # 3. theta_pi = theta_pi - train_lr * grad
                phi_i = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, phi_i)))

                logits_q = self.net(x_qry[i], phi_i, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
            
            # 1. Evaluate the train loss of mu_theta
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            mu_theta_tr = list(map(lambda p: p[0] - torch.exp(0.5 * p[1]) * p[2],
                                   zip(self.net.parameters(), self.log_gamma_p, grad)))
            
            # 2. Compute the KL divergence between theta|tr and theta|test
            loc_q = torch.cat([m.flatten() for m in mu_theta_test])
            scale_q = torch.cat([torch.exp(0.5 * v.flatten()) for v in self.log_v_q])
            # print(self.log_v_q[0])
            q_theta_test = torch.distributions.normal.Normal(loc_q, scale_q)
            
            loc_p = torch.cat([m.flatten() for m in mu_theta_tr])
            scale_p = torch.cat([torch.exp(0.5 * s.flatten()) for s in self.log_sigma2])
            # print(scale_p[0])
            # print(self.log_sigma2[0])
            p_theta_tr = torch.distributions.normal.Normal(loc_p, scale_p)
            
            kl = torch.distributions.kl.kl_divergence(q_theta_test, p_theta_tr).sum()
            # print(kl)
            kls += kl
            
            
        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        kl = kls / task_num
        # print(loss_q, kl)
        loss_all = loss_q + kl
        print('Loss:', loss_q.item(), kl.item(), end='\r')

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_all.backward()
        
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()
        
#         high = 1
#         low = 0
# #         log_v_q_max = torch.max(torch.cat([v.flatten() for v in self.log_v_q]))
# #         log_v_q_min = torch.min(torch.cat([v.flatten() for v in self.log_v_q]))
#         self.log_v_q = nn.ParameterList(
#             [torch.minimum(v, torch.ones_like(v, dtype=float) * 1e-5) for v in self.log_v_q]
#         )
        
# #         log_sigma2_max = torch.max(torch.cat([v.flatten() for v in self.log_sigma2]))
# #         log_sigma2_min = torch.min(torch.cat([v.flatten() for v in self.log_sigma2]))
#         self.log_sigma2 = nn.ParameterList(
#             [torch.minimum(s, torch.ones_like(s, dtype=float) * 1e-5) for s in self.log_sigma2]
#         )


        accs = np.array(corrects) / (querysz * task_num)
        # print(self.log_gamma_p[0])

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        
        # 1. Evaluate the train loss of mu_theta
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        mu_theta_tr = list(map(lambda p: p[0] - torch.exp(0.5 * p[1]) * p[2],
                               zip(net.parameters(), self.log_gamma_p, grad)))
        theta = mu_theta_tr
#         theta = list(map(lambda p: p[0] + torch.exp(0.5 * p[1]) * torch.randn(p[1].shape).type_as(p[1]),
#                          zip(mu_theta_tr, self.log_sigma2)))

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt, theta)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, theta)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, theta)))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, theta, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs




def main():
    pass


if __name__ == '__main__':
    main()
