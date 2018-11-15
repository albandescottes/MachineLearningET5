
from scipy.io import loadmat
from scipy.spatial import distance 
from random import randint
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import time


def load_data(path):
	data = loadmat(path)
	#return data
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


#cette méthode modifie la forme des données 
#pour être utilisées plus facilement dans le reste du programme
def simplifyValues(dataX, dataY):
	newDataX = []
	newDataY = []
	for i in range(0, dataX.shape[3]):
		newDataX.append(dataX[:,:,:,i])
		newDataY.append(dataY[i])
	return (np.array(newDataX), np.array(newDataY))

# DMIN
#NEW
#cette méthode renvoit le classifieur d'un jeu de données avec résultat
#avec un nombre de données passé en paramètre
def classifieurDMIN(dataX, dataY, number):
	classes = []
	finalClasses = []
	for i in range(0,10):
		classes.append([])
	for i in range(0 ,number):
		ind = dataY[i][0]
		if(ind == 10):
			classes[0].append(dataX[i])
		else:
			classes[ind].append(dataX[i])
	for i in range(0,10):
		finalClasses.append(np.mean(classes[i], axis=0))
	return finalClasses


# NEW
#cette méthode calcule pour chaque élément la classe qui a la distance euclidienne minimum
#elle retourne le pourcentage d'echec de l'échantillon avec le classifieur
def testDMIN(classes, dataX, dataY, number):
	if(np.array_equal(dataX[0].shape, classes[0].shape)):
		succes = 0
		fail = 0
		for i in range(0, number):
			bestValue = np.linalg.norm(classes[0]-dataX[i])
			bestMatch = 0
			for c in range(1, 10):
				temp = np.linalg.norm(classes[c]-dataX[i])
				if(temp < bestValue):
					bestValue = temp
					bestMatch = c
			y = dataY[i]
			if(y==10):
				y = 0
			if(bestMatch == y):
				succes += 1
			else:
				fail += 1
		return (fail / (succes+fail) * 100.)  
	else:
		print(dataX[0].shape, " != ", classes[0].shape)
		return 0;

#PCA
#cette méthode transforme les données en appelant la méthode PCA
#de la biblitthèque sklearn
def ACP_transformation(data,number, pca_values=10):
	dataFlatten = []
	for i in range(0,number):
		dataFlatten.append(data[i].flatten())
	pca = PCA(n_components=pca_values)
	return pca.fit_transform(dataFlatten)

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
#NEW MAIN
start_time = time.time()
train_values_X, train_values_y = load_data('train_32x32.mat')
test_values_X, test_values_y = load_data('test_32x32.mat')

train_x_after, train_y_after = simplifyValues(train_values_X, train_values_y)
sizeTrain = train_x_after.shape[0] - 1
test_x_after, test_y_after = simplifyValues(test_values_X, test_values_y)
sizeTest = test_x_after.shape[0] - 1
del train_values_X, train_values_y, test_values_y, test_values_X


#test DMIN avec toutes les valeurs des données train et test
classesDMIN = classifieurDMIN(train_x_after, train_y_after, sizeTrain)
print("DMIN with ", sizeTest , " values, percentage failed : " , testDMIN(classesDMIN, test_x_after, test_y_after, sizeTest), "%")



'''
#tests préprocessing d'Elodie
#sans pre procc
classesDMIN = classifieurDMIN(train_x_after, train_y_after, sizeTrain)
print("DMIN with ", sizeTest , " values, w/o preprocessing, percentage failed : " , testDMIN(classesDMIN, test_x_after, test_y_after, sizeTest), "%")
#pre procc
train_x_greyscale = rgb2gray(train_x_after).astype(np.float32)
test_x_greyscale = rgb2gray(test_x_after).astype(np.float32)
#avec pre procc
classesDMIN = classifieurDMIN(train_x_greyscale, train_y_after, sizeTrain)
print("DMIN with ", sizeTest , " values, w/ preprocessing, percentage failed : " , testDMIN(classesDMIN, test_x_greyscale, test_y_after, sizeTest), "%")
'''

'''
# tests avec différentes dimensions de l'ACP : 20 - 15 - 10
#20
data_ACP_train = ACP_transformation(train_x_after, 30000, 20)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_after, 10000, 20)
print("ACP -20- + DMIN with ", 10000, " values, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
#15
data_ACP_train = ACP_transformation(train_x_after, 30000, 15)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_after, 10000, 15)
print("ACP -15- + DMIN with ", 10000, " values, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
#10
data_ACP_train = ACP_transformation(train_x_after, 30000, 10)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_after, 10000, 10)
print("ACP -10- + DMIN with ", 10000, " values, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
'''

'''
# tests avec différentes dimensions de l'ACP : 20 - 15 - 10 
# plus preproccessing d'Elodie
#20
data_ACP_train = ACP_transformation(train_x_after, 30000, 20)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_after, 10000, 20)
print("ACP -20- + DMIN with ", 10000, " values, w/o preproccesing, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
#15
data_ACP_train = ACP_transformation(train_x_after, 30000, 15)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_after, 10000, 15)
print("ACP -15- + DMIN with ", 10000, " values, w/o preproccesing, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
#10
data_ACP_train = ACP_transformation(train_x_after, 30000, 10)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_after, 10000, 10)
print("ACP -10- + DMIN with ", 10000, " values, w/o preproccesing, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
train_x_greyscale = rgb2gray(train_x_after).astype(np.float32)
test_x_greyscale = rgb2gray(test_x_after).astype(np.float32)
#20
data_ACP_train = ACP_transformation(train_x_greyscale, 30000, 20)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_greyscale, 10000, 20)
print("ACP -20- + DMIN with ", 10000, " values, w/ preproccesing, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
#15
data_ACP_train = ACP_transformation(train_x_greyscale, 30000, 15)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_greyscale, 10000, 15)
print("ACP -15- + DMIN with ", 10000, " values, w/ preproccesing, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
#10
data_ACP_train = ACP_transformation(train_x_greyscale, 30000, 10)
classes_for_ACP_DMIN = classifieurDMIN(data_ACP_train, train_y_after, 30000)
data_ACP_test = ACP_transformation(test_x_greyscale, 10000, 10)
print("ACP -10- + DMIN with ", 10000, " values, w/ preproccesing, percentage failed : " , testDMIN(classes_for_ACP_DMIN, data_ACP_test, test_y_after, 10000), "%")
'''

print("--- %s seconds ---" % (time.time() - start_time))
#(width, height, channels, size)
#print("Training Set", X_train.shape, y_train.shape) 
#print("Test Set", X_test.shape, y_test.shape)

#Total number of images
#num_images = X_train.shape[0] + X_test.shape[0]
#print("Total Number of Images", num_images)

#displayAllImages(X_train, y_train, 2, 8)

#-----------Convert to GREY

#Transpose the image arrays (width, height, channels, size) -> (size, width, height, channels)
#X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
#X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

#Converting to Float for numpy computation
#rgb2gray(X_train)
#train_greyscale = rgb2gray(X_train).astype(np.float32)
#test_greyscale = rgb2gray(X_test).astype(np.float32)
#print("Training Set", train_greyscale.shape)
#print("Test Set", test_greyscale.shape)

#Removing RGB train, test, val set to reduce RAM Storage occupied by them
#del X_train, X_test
#image = train_greyscale[0]
#moyenneLignes(image)

#displayImage(train_greyscale, 1)
#displayAllImages(train_greyscale, y_train, 1, 10)

print("End of program")

















