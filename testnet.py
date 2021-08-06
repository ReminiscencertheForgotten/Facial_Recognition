from network import*
from dataset import*

def test_network(network, testloader):
    
    network.eval()
    print('Start testing...')
    test_num_correct = 0
    
    for batch in testloader:
        images, labels = batch
        
        preds = network(images)
        
        test_num_correct += preds.argmax(dim=1).eq(labels).sum().item()
    
    accuracy = test_num_correct / len(testloader.dataset)
    
    print(
        'accuracy: %.3f%%'%(accuracy * 100), '\n'
    )
    print('Test done.')
    
    accuracy = round(accuracy, 3)
    return accuracy