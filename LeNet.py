import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import matplotlib.pyplot as plt

torch.manual_seed(0)

def drawGraph(x, y):
    plt.title("Evolution de la précision")
    plt.plot(x, y)
    plt.xlabel('Epoch n°')
    plt.ylabel('Pourcentage de bonnes prédictions')
    plt.show()

class LeNet(nn.Module):
    #INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC => softmax
    def __init__(self):
            super(LeNet, self).__init__()

            #----Applies convolution
            #conv2d(in_channels, out_channels, kernel_size, stride, padding)

            #Couche 1: 6 filtres de convolution de taille 5 × 5 + ReLU + max pooling 2 × 2
            self.conv1 = nn.Conv2d(3, 6, 5)
            #Couche 2: 16 filtres de convolution de taille 5 × 5 + ReLU + max pooling 2 × 2
            self.conv2 = nn.Conv2d(6, 16, 5)

            #----Multiply inputs by learned weights
                #Linear(size of each input sample, size of each output sample)
            #Couche3 fully connected: 120 neurones + ReLU. (400 = 16*5*5)
            self.fc1 = nn.Linear(400, 120)
            #Couche4 fully connected: 84 neurones + ReLU.
            self.fc2 = nn.Linear(120, 84)
            #Couche5 fully connected: 10 neurones + log-softma
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
            #----Computes the activation of the first convolution
                #max_pool2d(kernel_size, stride, padding)
            x = F.relu(F.max_pool2d(self.conv1(x), 2)) 
            x = F.relu(F.max_pool2d(self.conv2(x), 2)) 

            #----Reshape data to input to the input layer of the neural net, -1 infers this dimension from the other given dimension
            x = x.view(x.shape[0], -1) # Flatten the tensor

            x = F.relu(self.fc1(x)) 
            x = F.relu(self.fc2(x)) 

            #----La perte « Softmax »est utilisée pour prédire une seule classe parmi K classes mutuellement exclusives
            x = F.log_softmax(self.fc3(x), dim=1)

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
    batch_size = 11 #avant: 10
    learning_rate = 0.001

    net = LeNet()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

#NEW
    train_size = train_label.size(0)
    test_size = test_label.size(0)
    print_every = 4 // 10
    epoch = []
    accuracy_train = []
    accuracy_test = []

    for e in range(epoch_nbr):
        print("Epoch", e)

        #NEW
        correct_train = 0
        correct_test = 0

        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(train_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_train, 1) #gives a tensor
            loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update

            #NEW - Stats Train par intervalle
            '''size = train_data[i:i+batch_size].size(0)
            correct_train += ((class_predicted == train_label[i:i+batch_size]).type(torch.int32)).sum().item() #number of correct predictions

            if (i + 1) % (print_every + 1) == 0:
                print('Train Epoch [{}/{}], Data [{}/{}], Total accuracy: {:.2f}%'
                    .format(e + 1, epoch_nbr, i+1, train_size, (correct_train / train_size) * 100))
            '''

            #NEW - Stats Test par intervalle
            '''predictions_test = net(test_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_test, 1) #gives a tensor
            size = test_data[i:i+batch_size].size(0)
            correct_test += ((class_predicted == test_label[i:i+batch_size]).type(torch.int32)).sum().item() #number of correct predictions

            if (i + 1) % (print_every + 1) == 0:
                print('Test Epoch [{}/{}], Data [{}/{}], Total accuracy: {:.2f}%'
                    .format(e + 1, epoch_nbr, i+1, test_size, (correct_test / test_size) * 100))
            '''

        epoch.append(e)
        #NEW - Stats Train total
        #print (accuracy_train)
        predictions_train = net(train_data[0:train_size])
        _, class_predicted = torch.max(predictions_train, 1) #gives a tensor
        correct_train = ((class_predicted == train_label).type(torch.int32)).sum().item() #number of correct predictions
        accuracy_train.append(correct_train/train_size*100)
        print("Score train: " + str(correct_train/train_size*100) + "%")             

        #NEW - Stats Test total
        predictions_test = net(test_data[0:test_size])
        _, class_predicted = torch.max(predictions_test, 1) #gives a tensor   
        correct_test = ((class_predicted == test_label).type(torch.int32)).sum().item() #number of correct predictions 
        accuracy_test.append(correct_test/test_size*100)         
        print("Score test: " + str(correct_test/test_size*100) + "%")   
        
#drawGraph(epoch,accuracy_train) 
#drawGraph(epoch,accuracy_test)    










