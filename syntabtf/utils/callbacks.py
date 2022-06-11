import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, mode='min', delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            mode (str): If 'min', the optimal value is minimum. If 'max', the optimal value is maximum
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: None
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode
    def __call__(self, val_input, model):

        score = val_input

        if self.best_score is None:
            self.best_score = score
            if self.path is not None:
                self.save_checkpoint(val_input, model)
        elif score < self.best_score + self.delta and self.mode == 'max':
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} (Finding maximum)')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score > self.best_score + self.delta and self.mode == 'min':
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} (Finding minimum)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.path is not None:
                self.save_checkpoint(val_input, model)
            self.counter = 0

    def save_checkpoint(self, val_input, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_input:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_input