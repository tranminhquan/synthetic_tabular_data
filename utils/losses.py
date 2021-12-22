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
            if span_info.activation_fn != 'softmax': #num value
                end = start + span_info.dim
                std = sigmas[start]
                eq = x[:, start] - torch.tanh(x_rec[:, start])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size(0))

                start = end
            else: # categorical values
                """
                QS: The categorical values are encoded using frequency, so how to calculate the loss of them.
                Should we use MSE or CE? 
                """ 
                
                end = start + span_info.dim
#                 loss.append(cross_entropy(x_rec[:, start:end], x[:, start:end], reduction='sum'))
                loss.append(mse_loss(x_rec[:, start:end], x[:, start:end], reduction='sum'))
                start = end
    print(start, x_rec.size(1))
#     assert start == x_rec.size()[1]
    
    # KL
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    
    return (sum(loss) * factor) / x.size(0), KLD / x.size(0)
    