from dataset import*
from train import*
from load import*
from network import*
from testnet import*
import warnings
warnings.filterwarnings('ignore')

systemparams = {
    'train_rate' : 0.4,
    'valid_rate' : 0.5,
    'src_filename' : 'Datafile/label.csv',
    'train_filename' : 'Datafile/train.csv',
    'valid_filename' : 'Datafile/valid.csv',
    'test_filename' : 'Datafile/test.csv',
    'root_path' : 'Dataset',
}

batch_size = 15
lr = 0.0002
patience = 15
n_epochs = 60

if __name__ == '__main__':
    
    load_Local_Data(params=systemparams)
    
    train_loader, valid_loader, test_loader = getDataloader(bacth_size=batch_size, params=systemparams)
    
    network = Network()
    
    network = train_network(
                network=network,
                trainloader=train_loader,
                validloader=valid_loader,
                learning_rate=lr,
                patience=patience,
                n_epochs=n_epochs
            )

    accuracy = test_network(network, test_loader)
    
    print('Saving the network...\n')
    torch.save(network.state_dict(), 'TrainResult/Model' + str(accuracy) + '.pth')
    
    print('All finished.')