from tqdm import tqdm

from syntabtf.utils.callbacks import EarlyStopping

def train(model, train_loader, epochs, optimizer, criterion, device, val_loader=None, hist=[], use_early_stopping=True, **kwargs):
    """
    Return:
        model
        history
    """
    
    model = model.to(device)
    output_info_list = kwargs['output_info_list']
    recloss_factor = kwargs['recloss_factor'] if 'recloss_factor'  in kwargs else 1.
    optimize_signature = kwargs['optimize_signature'] if 'optimize_signature' in kwargs else False
    
    if use_early_stopping:
        patience = kwargs['patience'] if 'patience' in kwargs else 7
        mode = kwargs['mode'] if 'mode' in kwargs else 'min'
        delta = kwargs['delta'] if 'delta' in kwargs else 0.
        early_stopping = EarlyStopping(patience=patience, verbose=True, mode=mode, delta=delta)

    for epoch in range(epochs):  # loop over the dataset multiple times
    
        print('Epoch: ', epoch + 1)
        running_loss = 0.0
        running_acc = 0.0
        processed = 0.0

        # TRAIN
        model.train()
        pbar = tqdm(train_loader, position=0)
        for i, data in enumerate(pbar):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0]
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, mu, logvar, sigmas = model(inputs)
            rec_loss, kl_loss = criterion(outputs, inputs, sigmas, mu, logvar, output_info_list, recloss_factor, optimize_signature)
            loss = rec_loss + kl_loss
                
            loss.backward()
            optimizer.step()
            
            # prunning
            model.decoder.sigmas.data.clamp_(0.01, 1.0)
            
            # print statistics
            running_loss += (loss.item() * inputs.size(0))
            processed += len(inputs)
            
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={i}')
            
    #         pbar.set_description(desc= f'Loss={loss.item()} Batch_id={i} Accuracy={acc.item() / inputs.size(0)}')

        print('- Avg.loss: %.4f' % (running_loss / len(train_loader.dataset)))
        avgloss = running_loss / len(train_loader.dataset)
        
        # early stopping
        if use_early_stopping:
            early_stopping(avgloss, model)
            if early_stopping.early_stop:
                print("*** Early stopping ***")
                break

        hist.append([avgloss])

    return model, hist


