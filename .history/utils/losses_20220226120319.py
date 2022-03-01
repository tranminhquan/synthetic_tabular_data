import torch
from torch.nn.functional import cross_entropy, mse_loss

def vae_loss(x, x_rec, sigmas, mu, logvar, output_info_list, factor=1):
    """
    Loss of VAE: reconstruction loss and KDL
    """
    start = 0
    loss = []
    
    for output_info in output_info_list:
        for span_info in output_info:
            if span_info.activation_fn != 'softmax': #numerical values (alpha)
                end = start + span_info.dim
                std = sigmas[start]
#                 print('x rec numerical: ', x[:, start].size())
#                 print('std: ', std.size())
                eq = x[:, start] - torch.tanh(x_rec[:, start])
                loss.append((eq ** 2 / (2 * (std ** 2))).sum())
#                 print('numerical loss with std: ', loss[-1])
                
                loss.append(torch.log(1./std) * x.size(0))
#                 print('log loss: ', (torch.log(1./std) * x.size()[0]))
#                 print('x size at 0: ', x.size()[0])
#                 print('log std: ', loss[-1])
                start = end
            else: # categorical values (beta and d)
                """
                QS: The categorical values are encoded using frequency, so how to calculate the loss of them.
                Should we use MSE or CE? 
                """ 
                
                end = start + span_info.dim
#                 loss.append(cross_entropy(x_rec[:, start:end], x[:, start:end], reduction='sum'))
#                 print('x_rec without act_fn: ', x_rec[:, start:end])
    
                # experiment: add sigmoid
#                 print('x_rec cate: ', x_rec[:, start:end].size())
#                 loss.append(mse_loss(torch.sigmoid(x_rec[:, start:end]), x[:, start:end], reduction='sum'))
                loss.append(mse_loss((x_rec[:, start:end]), x[:, start:end], reduction='sum'))
#                 print('mse categorical loss: ', loss[-1])
                start = end
    
    # KL
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    
    print('vae loss: ', (sum(loss) * factor) / x.size(0), ' | KLD: ', KLD / x.size(0)), 
    return (sum(loss) * factor) / x.size(0), KLD / x.size(0)
    