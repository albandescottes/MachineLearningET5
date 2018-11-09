
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
	data = loadmat(path)
	return data['X'], data['y']
	#train_data.items()

def displayImage(matrice, index):
	print('Image n°: ', matrice['y'][index])
	plt.imshow(matrice['X'][:, 1:, :, index])
	plt.show()

def displayAllImages(img, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
    plt.show()    

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

'''
1. Nettoyer les données
idées: Normalising the intensity, global and local contrast normalisation, ZCA whitening

2. Classifieur à distance minimum (only numpy & matplotlib)
	- Déterminer pour chaque chiffre: vecteur de caractéristiques
	- Calculer le centre des vecteurs de chaque classe (0 à 9)
	- Sauvegarder dans un fichier d'apprentissage
	- Ecrire le programme de décision:
		- Déterminer pour chaque chiffre: vecteur des distances 
		- Classer en comparant au centre de chaque classe


3. Réduction de la dimension des vecteurs (ACP)
'''

#-----------Reading the .MAT files

X_train, y_train = load_data('train_32x32.mat')
X_test, y_test = load_data('test_32x32.mat')

#(width, height, channels, size)
#print("Training Set", X_train.shape, y_train.shape) 
#print("Test Set", X_test.shape, y_test.shape)

#Total number of images
num_images = X_train.shape[0] + X_test.shape[0]
#print("Total Number of Images", num_images)

#displayAllImages(X_train, y_train, 2, 8)

#-----------Convert to GREY

#Transpose the image arrays (width, height, channels, size) -> (size, width, height, channels)
X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

#Converting to Float for numpy computation
rgb2gray(X_train)
train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)
#print("Training Set", train_greyscale.shape)
#print("Test Set", test_greyscale.shape)

#Removing RGB train, test, val set to reduce RAM Storage occupied by them
del X_train, X_test
displayAllImages(train_greyscale, y_train, 1, 10)

print("End of program")

















