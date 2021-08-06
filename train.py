from pytorchtool import EarlyStopping
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
from progressbar import*

def train_network(network, trainloader, validloader, learning_rate, patience, n_epochs):
    
    lr = learning_rate
    estop = EarlyStopping(patience=patience, verbose=True)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    widgets = [
        'Progress:', Bar('#'), Percentage()
    ]
    train_total = len(trainloader.dataset) // trainloader.batch_size + 1
    
    for epoch in range(1, n_epochs + 1):
        
        print(
            'epoch', epoch,
            'start training...'
        )
        
        network.train()
        epoch_num_correct = 0
        
        pbar = ProgressBar(widgets=widgets).start()
        i = 1
        
        for batch in trainloader:
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.update(int((i / (train_total - 1) * 100)))
            i += 1
            
        pbar.finish()
        
        print(
            'epoch', epoch,
            'train done,start validing...'
        )
        
        network.eval()
        for batch in validloader:
            images, labels = batch
            
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            
            epoch_num_correct += preds.argmax(dim=1).eq(labels).sum().item()

            valid_losses.append(loss.item())
        
        accuracy = epoch_num_correct / len(validloader.dataset)
        print(
            'epoch', epoch,
            'valid done.'
        )      
 
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        print(
            f'train_loss: {train_loss:.5f}', 
            f'valid_loss: {valid_loss:.5f}',
            'valid_accuracy: %.3f%%'%(accuracy * 100),
        )
        
        train_losses = []
        valid_losses = []
        
        estop(valid_loss, network)
        print('\n')
        
        if estop.early_stop:
            print('Stop to train network.\n')
            break
    
    print('Epoch finished,network reloading...')
    network.load_state_dict(torch.load('checkpoint.pt'))
    print('Network reloaded.\n')   
    return network