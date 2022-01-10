from tqdm import tqdm

def train(model, train_loader, epochs, optimizer, criterion, device, val_loader=None, hist=[], **kwargs):
    """
    Return:
        model
        history
    """
    
    model = model.to(device)
    output_info_list = kwargs['output_info_list']
    recloss_factor = kwargs['recloss_factor'] if 'recloss_factor' in kwargs else 1.

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
            rec_loss, kl_loss = criterion(outputs, inputs, sigmas, mu, logvar, output_info_list, recloss_factor)
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

        hist.append([avgloss])

    return model, hist