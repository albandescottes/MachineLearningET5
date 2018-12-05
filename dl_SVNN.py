import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import time
torch.manual_seed(0)

class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__()

            #Applies convolution
                #conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv1 = nn.Conv2d(3, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)

            #Multiply inputs by learned weights
                #fc: couche entièrement connectée
                #Linear(size of each input sample, size of each output sample)
            self.fc1 = nn.Linear(1250, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
            #Computes the activation of the first convolution
                #max_pool2d(kernel_size, stride, padding)
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))

            #Reshape data to input to the input layer of the neural net
            #-1 infers this dimension from the other given dimension
            x = x.view(x.shape[0], -1) # Flatten the tensor

            #Computes the activation of the first fully connected layer
            x = F.relu(self.fc1(x))

            #Couche de perte
            #Computes the second fully connected layer (activation applied later)
            #La perte « Softmax »est utilisée pour prédire une seule classe parmi K classes mutuellement exclusives
            x = F.log_softmax(self.fc2(x), dim=1)

            return x


if __name__ == '__main__':

    # Load the dataset
    train_data = loadmat('train_32x32.mat')
    test_data = loadmat('test_32x32.mat')

    train_label = train_data['y'][:100]
    train_label = np.where(train_label==10, 0, train_label)
    train_label = torch.from_numpy(train_label.astype('int')).squeeze(1)
    train_data = torch.from_numpy(train_data['X'].astype('float32')).permute(3, 2, 0, 1)[:100]

    test_label = test_data['y'][:1000]
    test_label = np.where(test_label==10, 0, test_label)
    test_label = torch.from_numpy(test_label.astype('int')).squeeze(1)
    test_data = torch.from_numpy(test_data['X'].astype('float32')).permute(3, 2, 0, 1)[:1000]

    # Hyperparameters
    epoch_nbr = 10
    batch_size = 12 #avant: 10
    learning_rate = 0.001

    net = CNN()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    #NEW
    n_batches = 4

    for e in range(epoch_nbr):
        print("Epoch", e)

        #NEW
        print_every =  n_batches // 10
        train_size = train_label.size(0)
        test_size = test_label.size(0)

        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(train_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_train, 1) #gives a tensor
            loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update

            #NEW - Stats Train par intervalle
            size = train_data[i:i+batch_size].size(0)
            correct_train = ((class_predicted == train_label[i:i+batch_size]).type(torch.int32)).sum().item() #number of correct predictions

            if (i + 1) % (print_every + 1) == 0:
                print('Train Epoch [{}/{}], Data [{}/{}], Total accuracy: {:.2f}%'
                    .format(e + 1, epoch_nbr, i+1, train_size, (correct_train / size) * 100))
            
            #NEW - Stats Test par intervalle
            predictions_test = net(test_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_test, 1) #gives a tensor
            size = test_data[i:i+batch_size].size(0)
            correct_test = ((class_predicted == test_label[i:i+batch_size]).type(torch.int32)).sum().item() #number of correct predictions

            if (i + 1) % (print_every + 1) == 0:
                print('Test Epoch [{}/{}], Data [{}/{}], Total accuracy: {:.2f}%'
                    .format(e + 1, epoch_nbr, i+1, test_size, (correct_test / size) * 100))
            

        #NEW - Stats Train total
        #print (accuracy_train)
        predictions_train = net(train_data[0:train_size])
        _, class_predicted = torch.max(predictions_train, 1) #gives a tensor
        correct_train = ((class_predicted == train_label).type(torch.int32)).sum().item() #number of correct predictions
        print("Score train: " + str(correct_train/train_size*100) + "%")              
          

        #NEW - Stats Test total
        predictions_test = net(test_data[0:test_size])
        _, class_predicted = torch.max(predictions_test, 1) #gives a tensor   
        correct_train = ((class_predicted == test_label).type(torch.int32)).sum().item() #number of correct predictions          
        print("Score test: " + str(correct_train/test_size*100) + "%")   
        




