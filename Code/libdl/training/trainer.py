import os
import sys
import copy
basepath = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(basepath)
import torch as t
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import logging
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self,
                 model,
                 crit,
                 optim=None,
                 lr_schedule=None,
                 early_stop=None,
                 train_dl=None,
                 val_dl=None,
                 device="cuda:0",
                 path_trainer_ckp=None,
                 ckp_prefix=None,
                 clamp_params=None,
                 l1_params=None,
                 tracked_params=None,
                 only_save_best=False,
                 verbose=True,
                 logfile=None,
                 split_loss_weakLabels = False,
                 plot_e_matrix=False,
                 plot_e_matrix_params=None,
                 plot_loader=None,
                 plot_loader_strong=None,
                 gamma_schedule = None,
                 lr_scheduler_start=0,
                 D_prior_params=None,
                 ):
        """ Train a neural network model (torch.nn.Module)

        Args:
            model: model to be trained
            crit: loss function
            optim: optimizer to be used
            lr_schedule: learning rate scheduler
            early_stop: early stopping
            train_dl: training data loader
            val_dl: validation data loader
            device: device to use for training (default: 'cuda:0')
            path_trainer_ckp: path where to store checkpoints and model state dicts
            ckp_prefix: prefix for checkpoint and model files
            clamp_params: dictionary of nodes and the respective allowed intervals for constrained optimization
                (e.g. {'conv.weight': [0, None]} restricts the weights of module 'conv' to be in the interval [0, inf])
            l1_params: dictionary of nodes and the respective regularization weights, (e.g. {'conv.weight': 1e-4} adds
                penalty of 1e-4 * sum(abs(weights, biases)) to loss)
            tracked_params: list of parameter names whose history is recorded during training;
                accessible via property Trainer.tracked_parameters
            only_save_best: whether to only save best trainer ckp / model and thus overwrite previous best states
            verbose: whether to print training stats (default: True)
        """
        self._model = model.to(device)
        self._crit = crit.to(device)
        self._optim = optim
        self._lr_schedule = lr_schedule
        self._early_stop = early_stop
        self._train_dl = train_dl
        self._val_dl = val_dl
        self._device = device
        self.path_trainer_ckp = path_trainer_ckp
        self.ckp_prefix = ckp_prefix
        self._clamp_params = clamp_params
        self._l1_params = l1_params
        self._tracked_params = tracked_params
        self._only_save_best = only_save_best
        self.verbose = verbose
        
        self.split_loss_weakLabels = split_loss_weakLabels
        

        self._train_losses, self._val_losses = [], []

        if isinstance(tracked_params, list) and len(tracked_params) > 0:
            self._param_history = {}
            for name in tracked_params:
                self._param_history[name] = []
                
        self.logfile = logfile
        if self.logfile is not None:
            with open(self.logfile, "a") as lf:
                lf.write("epoch;trainLoss;valLoss;lr;")
                
        
        self.plot_e_matrix = plot_e_matrix
        self.plot_e_matrix_params = plot_e_matrix_params
        self.plot_loader = plot_loader
        self.plot_loader_strong = plot_loader_strong
        self.lr_scheduler_start = lr_scheduler_start
        
        self.gs=gamma_schedule
        
        self.Dpp = D_prior_params
        self.curr_epoch = 0
        self.prior_weight=0
    
    # construct diagonal prior matrix
    def get_diag_prior(self, N, M, nu=1e4):
        diagPath = np.linspace(0, M, N, endpoint=False).astype(int)

        frame_durations = np.zeros(M)
        for n in diagPath:
            frame_durations[n]+=1
        D_prior = np.zeros((M, N))
        curr_fr_start = 0
        curr_fr_end = 0
        for fr, dur in enumerate(frame_durations.astype(int)):
            curr_fr_end = curr_fr_start + dur
            D_prior[fr, curr_fr_start : curr_fr_end] = 1
            D_prior[fr, :curr_fr_start] = np.exp(- (np.arange(curr_fr_start)-curr_fr_start)**2 / (2*nu))
            D_prior[fr, curr_fr_end:] = np.exp(- (np.arange(curr_fr_end, N)-curr_fr_end)**2 / (2*nu))
            curr_fr_start = curr_fr_end
        return 1-D_prior.transpose()
    
    # plot soft alignment ("E-matrix") after each training epoch
    def plot_e_mtrx(self):        
        x = self.plot_e_matrix_params["x"]
        y = self.plot_e_matrix_params["y"]
        lastChange = self.plot_e_matrix_params["lastChange"]
        opt_paths = self.plot_e_matrix_params["opt_paths"]
        
        self._model.zero_grad()
        
        fig, ax = plt.subplots(4, 4, figsize=(20, 10))
        ax = ax.flatten()

        for b in range(y.shape[0]):
            # forward pass
            y_pred = self._model(x[b:b+1,:])
            
            # get D prior
            if self.Dpp is not None:
                D_prior = torch.Tensor(self.get_diag_prior(y_pred.shape[2], lastChange[b].cpu()+1, nu=self.Dpp["nu"])[None,:]).to(self._device)
            else:
                D_prior = 0
            
            if self.Dpp is not None:
                loss_ = self._crit(y_pred, y[b:b+1, :, :lastChange[b]+1, :], D_prior=D_prior*self.prior_weight)
            else:
                loss_ = self._crit(y_pred, y[b:b+1, :, :lastChange[b]+1, :])
            
            # backward pass
            loss_.backward()

            ax[b].imshow(self._crit.dtw_class.e_matrix[0].cpu().detach().numpy().transpose(), aspect='auto', origin='lower', interpolation="none", cmap='gray_r')
            ax[b].plot(opt_paths[b][1], opt_paths[b][0], color='tab:red', linewidth=2)
    
        plt.show()
        
        # delete gradients 
        self._model.zero_grad()
        
        
        
        
                
    def save_checkpoint(self):
        """
        When called, saves a file "(ckp_prefix)_checkpoint_(total #epochs).ckp" in folder self.path_trainer_ckp.
        Can be used to continue training from a certain checkpoint.
        """
        epoch = len(self._train_losses)

        if self._only_save_best:
            fn_ckp = f'{self.ckp_prefix}_checkpoint.ckp'
        else:
            fn_ckp = f'{self.ckp_prefix}_checkpoint_{epoch:03d}.ckp'

        path_ckp = os.path.join(basepath, self.path_trainer_ckp, fn_ckp)

        t.save({
                'epoch': epoch,
                'model_state_dict': self._model.state_dict(),
                'optim_state_dict': self._optim.state_dict(),
                'lr_scheduler_state_dict': self._lr_schedule.state_dict(),
                'early_stopping_best': self._early_stop.best,
                'train_losses': self._train_losses,
                'val_losses': self._val_losses
                }, path_ckp)

        self.save_model_only()

    def save_model_only(self):
        """
        When called, saves a file "(ckp_prefix)_model_(total #epochs).pt" in folder self.path_trainer_ckp.
        """
        epoch = len(self._train_losses)

        if self._only_save_best:
            fn_model = f'{self.ckp_prefix}_model.pt'
        else:
            fn_model = f'{self.ckp_prefix}_model_{epoch:03d}.pt'

        path_model = os.path.join(basepath, self.path_trainer_ckp, fn_model)
        t.save(self._model.state_dict(), path_model)

    def restore_checkpoint(self, epoch):
        """
        Restores state dicts of model, optimizer, learning rate scheduler and loss lists from checkpoint.
        """
        if self._only_save_best:
            fn_ckp = f'{self.ckp_prefix}_checkpoint.ckp'
        else:
            fn_ckp = f'{self.ckp_prefix}_checkpoint_{epoch:03d}.ckp'

        path_ckp = os.path.join(basepath, self.path_trainer_ckp, fn_ckp)

        if os.path.exists(path_ckp):
            checkpoint = torch.load(path_ckp, map_location='cpu')

            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.to(self._device)
            self._optim.load_state_dict(checkpoint['optim_state_dict'])
            self._lr_schedule.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self._early_stop.best = checkpoint['early_stopping_best']
            self._train_losses = checkpoint['train_losses']
            self._val_losses = checkpoint['val_losses']

            self._model.eval()
            print(f'Loaded trainer state dicts of epoch {epoch}.')
        else:
            self.load_model(epoch)

    def load_model(self, epoch):
        """
        Load model state dict from a previously saved file in folder self.path_trainer_ckp.
        """
        if self._only_save_best:
            fn_model = f'{self.ckp_prefix}_model.pt'
        else:
            fn_model = f'{self.ckp_prefix}_model_{epoch:03d}.pt'

        path_model = os.path.join(basepath, self.path_trainer_ckp, fn_model)

        if os.path.exists(path_model):
            self._model.load_state_dict(torch.load(path_model, map_location=self._device))
            self._model.eval()
            print(f'Loaded model of epoch {epoch}.')
        else:
            raise ValueError(f'Could not find a saved model of epoch {epoch}!')

    def clamp_parameters(self):
        for name, param in self._model.named_parameters():
            if name in self._clamp_params:
                param.data = torch.clamp(param.data, min=self._clamp_params[name][0], max=self._clamp_params[name][1])

    def l1_penalty(self):
        parameter_list = []
        for name, param in self._model.named_parameters():
            if name in self._l1_params:
                parameter_list.append(self._l1_params[name] * param.view(-1))
        parameters = torch.cat(parameter_list)
        return F.l1_loss(parameters, torch.zeros_like(parameters), reduction='sum')

    def track_parameters(self):
        for name, param in self._model.named_parameters():
            if name in self._tracked_params:
                self._param_history[name].append(param.data.clone().to('cpu').detach())

    def train_step(self, x, y):
        """
        Trains the model with one mini batch.
        Inputs x, targets y.
        """
        self._model.zero_grad()
        y_pred = self._model(x)
                
        # weak label sequences generally have different lengths within a batch, so we need to find the last element belonging to the sequence and do seperate forward passes 
        if self.split_loss_weakLabels:
            changes = (y[:, :, 1:, :] != y[:, :, :-1, :]).any(axis=3)
            lastChange = torch.sum(changes, axis=[1,2])
            
            loss_batch = []
            for b in range(y.shape[0]):
                # get D prior
                if self.Dpp is not None:
                    D_prior = torch.Tensor(self.get_diag_prior(y_pred.shape[2], lastChange[b].cpu()+1, nu=self.Dpp["nu"])[None,:]).to(self._device)
                else:
                    D_prior = 0
                
                loss_batch.append(self._crit(y_pred[b:b+1], y[b:b+1, :, :lastChange[b]+1, :], D_prior=D_prior*self.prior_weight))
                
            loss = torch.mean(torch.stack(loss_batch))            
        else:
            if self.Dpp is not None:
                    D_prior = torch.Tensor(self.get_diag_prior(y_pred.shape[2], y_pred.shape[2], nu=self.Dpp["nu"])[None,:]).to(self._device)
                    loss = self._crit(y_pred, y, D_prior=D_prior*self.prior_weight)
            else:
                D_prior = 0
                loss = self._crit(y_pred, y)
            
        if self._l1_params is not None:
            loss += self.l1_penalty()
            
        # backward pass is now computed over all sequences within a batch
        loss.backward()
        self._optim.step()
        if self._clamp_params is not None:
            self.clamp_parameters()
        return loss.item()

    def val_test_step(self, x, y):
        """
        Performs validation / testing on a mini batch.
        """
        y_pred = self._model(x)
        if self.split_loss_weakLabels:
            changes = (y[:, :, 1:, :] != y[:, :, :-1, :]).any(axis=3)
            lastChange = torch.sum(changes, axis=[1,2])
            
            loss_batch = []
            for b in range(y.shape[0]):              
                # get D prior
                if self.Dpp is not None:
                    D_prior = torch.Tensor(self.get_diag_prior(y_pred.shape[2], lastChange[b].cpu()+1, nu=self.Dpp["nu"])[None,:]).to(self._device)
                    loss_batch.append(self._crit(y_pred[b:b+1], y[b:b+1, :, :lastChange[b]+1, :], D_prior=D_prior*self.prior_weight))
                else:
                    D_prior = 0                
                    loss_batch.append(self._crit(y_pred[b:b+1], y[b:b+1, :, :lastChange[b]+1, :]))
                
            loss = torch.mean(torch.stack(loss_batch))            
        else:
            loss = self._crit(y_pred, y)
        return loss.item(), y_pred

    def train_epoch(self):
        """
        Trains the model for one epoch using the training dataloader and returns the mean loss.
        """
        self._model.train()

        with t.enable_grad():
            loss_accum, n_batches = 0, 0

            for x, y in self._train_dl:
                # Transfer data to GPU if necessary
                x = x.to(self._device)
                y = y.to(self._device)
                loss = self.train_step(x, y)
                loss_accum += loss
                n_batches += 1

            train_loss = loss_accum / n_batches
            return train_loss

    def validate(self):
        """
        Computes and returns validation loss using the validation dataloader.
        """
        self._model.eval()

        with t.no_grad():
            loss_accum, n_batches = 0, 0

            for x, y in self._val_dl:
                # Transfer data to GPU if necessary
                x = x.to(self._device)
                y = y.to(self._device)
                loss, _ = self.val_test_step(x, y)
                loss_accum += loss

                n_batches += 1

            val_loss = loss_accum / n_batches
            return val_loss

    def test(self, test_dl):
        """
        Tests model performance on data provided by a test dataloader.
        Returns test loss, model predictions and targets.
        """
        y_targets = None
        y_preds = None

        self._model.eval()

        with t.no_grad():
            loss_accum, n_batches = 0, 0

            for x, y in test_dl:
                if y_targets is not None:
                    y_targets = t.cat((y_targets, y), dim=0)
                else:
                    y_targets = y

                # Transfer data to GPU if necessary
                x = x.to(self._device)
                y = y.to(self._device)

                loss, y_pred = self.val_test_step(x, y)
                loss_accum += loss

                if y_preds is not None:
                    y_preds = t.cat((y_preds, y_pred.cpu()), dim=0)
                else:
                    y_preds = y_pred.cpu()

                n_batches += 1

            test_loss = loss_accum / n_batches
            return test_loss, y_preds, y_targets

    def fit(self, epochs=1):
        """
        Trains the model for 'epochs' epochs. Returns list of training and validation losses.
        """
        best_model_epoch = 0

        for epoch in range(epochs):            
            if self.gs is not None:
                if epoch < self.gs["steps_const"]:
                    self._crit.gamma = self.gs["initial_gamma"]
                elif (epoch >= self.gs["steps_const"]) and (epoch < self.gs["steps_const"]+self.gs["steps_decay"]):
                    self._crit.gamma = self.gs["initial_gamma"] + (self.gs["final_gamma"] - self.gs["initial_gamma"])/self.gs["steps_decay"] * (epoch-self.gs["steps_const"])
                else:
                    self._crit.gamma = self.gs["final_gamma"]
                    
                print("current gamma: %.4f"%(self._crit.gamma))
                
                
            if self.Dpp is not None:
                if epoch < self.Dpp["steps_const"]:
                    self.prior_weight = self.Dpp["initial_weight"]
                elif (epoch >= self.Dpp["steps_const"]) and (epoch < self.Dpp["steps_const"]+self.Dpp["steps_decay"]):
                    self.prior_weight = self.Dpp["initial_weight"] + (self.Dpp["final_weight"] - self.Dpp["initial_weight"])/self.Dpp["steps_decay"] * (epoch-self.Dpp["steps_const"])
                else:
                    self.prior_weight = self.Dpp["final_weight"]
                    
                print("current prior weight: %.4f"%(self.prior_weight))
            
            train_loss = self.train_epoch()
            self._train_losses.append(train_loss)

            val_loss = self.validate()
            self._val_losses.append(val_loss)

            if self._tracked_params is not None:
                self.track_parameters()

            if self.verbose:
                print('Finished epoch ' + str(len(self._train_losses)) + '. Train Loss: ' + "{:.4f}".format(train_loss) +
                      ', Val Loss: ' + "{:.4f}".format(val_loss) + ' with lr: ' +
                      "{:.5f}".format(self._optim.param_groups[0]['lr']))
                
                
            if self.logfile is not None:
                with open(self.logfile, "a") as lf:
                    lf.write("\n%i;%.4f;%.4f;%.6f;"%(len(self._train_losses),
                                                   train_loss,
                                                   val_loss,
                                                   self._optim.param_groups[0]['lr']))

            if epoch < self.lr_scheduler_start:
                scheduler_loss = 100-epoch
            else:
                scheduler_loss = val_loss
            # Learning rate scheduler
            if self._lr_schedule is not None:
                self._lr_schedule.step(scheduler_loss)

            if not self._only_save_best:
                self.save_checkpoint()

            # Early stopping / saving
            if self._early_stop is not None:
                if len(self._train_losses) > 1 and self._early_stop.curr_is_better(scheduler_loss) or len(self._train_losses) == 1:
                    if self._only_save_best:
                        self.save_checkpoint()
                    best_model_epoch = len(self._train_losses)
                    if self.verbose:
                        print('  ... model of epoch {} saved'.format(len(self._train_losses)))

                if self._early_stop.step(scheduler_loss):
                    if self.verbose:
                        print('Early stopping applied!')
                    break
           
                
                    
            if self.plot_e_matrix:
                self.plot_e_mtrx()

        if self._early_stop is None or epochs == 0:
            self.save_checkpoint()

        if self._early_stop is not None:
            # Load best model
            self.load_model(best_model_epoch)

        return self._train_losses, self._val_losses

    @property
    def model(self):
        return copy.deepcopy(self._model)

    @model.setter
    def model(self, model):
        self._model = model.to(self._device)

    @property
    def tracked_parameters(self):
        return self._param_history
