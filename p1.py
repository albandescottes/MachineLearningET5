
from scipy.io import loadmat
from scipy.spatial import distance 
from random import randint
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import time

from skimage.filters import threshold_mean
from skimage import data, filters

#cette méthode charge la matrice du fichier .mat
def load_data(path):
	data = loadmat(path)
	#return data
	return data['X'], data['y']
	#train_data.items()

#cette méthode affiche l'image "index" de la matrice
def displayImage(matrice, index):
	print('Image n°: ', matrice['y'][index])
	plt.imshow(matrice['X'][:, 1:, :, index])
	plt.show()

#cette méthode affiche les nrows, ncols images de la matrice
def displayAllImages(img, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i], cmap="gray")
        else:
            ax.imshow(img[i,:,:,0], cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
    plt.show()    

#cette méthode transforme une matrice rgb en noir et blanc
def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

#OLD
#Preprocessing
#cette méthode applique un filtre de preprocessing à une matrice
def applyFilter(img, size):
	imgFiltered = [] 
	for i in range (0, size):
		mean = threshold_mean(img[i])
		#mid = img[i][16][16]

		hyst = filters.apply_hysteresis_threshold(img[i]-5, mean, mean+5)

		#hyst = filters.apply_hysteresis_threshold(img[i], mid-5, mid+5)
		imgFiltered.append(hyst)
	return np.array(imgFiltered)

#OLD
#cette méthode modifie la forme des données 
#pour être utilisées plus facilement dans le reste du programme
def simplifyValues(dataX, dataY):
	newDataX = []
	newDataY = []
	for i in range(0, dataX.shape[3]):
		newDataX.append(dataX[:,:,:,i])
		newDataY.append(dataY[i])
	return (np.array(newDataX), np.array(newDataY))

#OLD
#PCA
#cette méthode transforme les données en appelant la méthode PCA
#de la biblitthèque sklearn
def ACP_transformation(data,number, pca_values=10):
	dataFlatten = []
	for i in range(0,number):
		dataFlatten.append(data[i].flatten())
	pca = PCA(n_components=pca_values)
	return pca.fit_transform(dataFlatten)

#OLD
#SVM svc 
def flattenData(data):
	dataFlatten = []
	for i in range(0, data.shape[0]):
		dataFlatten.append(data[i].flatten())
	return dataFlatten

#NEW
#méthodes pour le classifieur DMIN
def DMIN_classifier(dataX, dataY):
	classes = []
	finalClasses = []
	for i in range(0,10):
		classes.append([])
	for i in range(0 ,dataY.shape[0]):
		classes[dataY[i]].append(dataX[i])
	for i in range(0,10):
		finalClasses.append(np.mean(classes[i], axis=0))
	return finalClasses

#NEW
def DMIN_predict(dataX, dmin):
	predicitons = []
	for i in range(0, len(dataX)):
		bestValue = np.linalg.norm(dmin[0]-dataX[i])
		bestMatch = 0
		for c in range(1, 10):
			temp = np.linalg.norm(dmin[c]-dataX[i])
			if(temp < bestValue):
				bestValue = temp
				bestMatch = c
		predicitons.append(bestMatch)
	return np.array(predicitons)

#NEW
#flatten and cut in x and y
def load_data_flatten(path, preprocessing=0):
	data = loadmat(path)
	data_x = []
	data_y = []
	for i in range(0, data['X'].shape[3]):
		if preprocessing == 0:
			data_x.append(data['X'][:,:,:,i].flatten())
		else:
			data_x.append(preprocessing_image(data['X'][:,:,:,i]).flatten())
		data_y.append(data['y'][i])
	data_y[data_y == 10] = 0
	data_y = np.array(data_y)
	data_y[data_y == 10] = 0
	return np.array(data_x), data_y

#NEW
def preprocessing_image(img):
	mean = threshold_mean(img)
	return filters.apply_hysteresis_threshold(img -5, mean, mean + 5)

#NEW
#acp
def acp_transformation(data, pca_values=10):
	pca = PCA(n_components=pca_values)
	return pca.fit_transform(data)

#NEW
#méthode DMIN
def dmin_system(trainX, trainY, testX, testY):
	start_time = time.time()
	print('----DMIN-------------')
	dmin = DMIN_classifier(trainX, trainY)
	dmin_predicitons = DMIN_predict(testX, dmin)
	accuracy, confusion_matrix = results_system(dmin_predicitons, testY)
	print('dmin accuracy = ', accuracy)
	print_confusion_matrix(confusion_matrix)
	print("execution time for DMIN %s seconds ---" % (time.time() - start_time))
	print('---------------------')

#NEW
#méthode SVM, utilisation de SVC
def svm_system(trainX, trainY, testX, testY):
	start_time = time.time()
	print('----SVM--------------')
	svc = SVC(kernel = 'linear')
	svc.fit(trainX, trainY)
	svc_predicitons = svc.predict(testX)
	accuracy, confusion_matrix = results_system(svc_predicitons, testY)
	print('svc accuracy = ', accuracy)
	print_confusion_matrix(confusion_matrix)
	print("execution time for svm %s seconds ---" % (time.time() - start_time))
	print('---------------------')

#NEW
#méthode Neighbors, utilisation de KNeighborsClassifier
def knn_system(trainX, trainY, testX, testY, k=10):
	start_time = time.time()
	print('----KNN--------------')
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(trainX,trainY)
	knn_prediction = knn.predict(testX)
	accuracy, confusion_matrix = results_system(knn_prediction, testY)
	print('knn accuracy = ', accuracy)
	print_confusion_matrix(confusion_matrix)
	print("execution time for knn %s seconds ---" % (time.time() - start_time))
	print('---------------------')

#NEW
#méthode qui calcule la précision du système ainsi que la matrice de confusion
def results_system(predictions, values):
	confusion_matrix = np.zeros((10,10))
	for i in range(0, predictions.shape[0]):
		confusion_matrix[values[i], predictions[i]] += 1
	#print(confusion_matrix)
	accuracy = np.trace(confusion_matrix) * 100 / predictions.shape[0]
	return accuracy, confusion_matrix

#NEW
#méthode qui affiche la matrice de confusion sur le terminal
def print_confusion_matrix(matrix):
	print('\t| 0\t| 1\t| 2\t| 3\t| 4\t| 5\t| 6\t| 7\t| 8\t| 9')
	for i in range(0,10):
		print('-------------------------------------------------------------------------------------')
		print('{}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}'.format(i, matrix[i,0].astype(np.int32), \
			matrix[i,1].astype(np.int32), matrix[i,2].astype(np.int32), matrix[i,3].astype(np.int32), matrix[i,4].astype(np.int32),\
			 matrix[i,5].astype(np.int32), matrix[i,6].astype(np.int32), matrix[i,7].astype(np.int32), matrix[i,8].astype(np.int32), \
			 matrix[i,9].astype(np.int32)))
		
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
'''
train_values_X, train_values_y = load_data('train_32x32.mat')
test_values_X, test_values_y = load_data('test_32x32.mat')

train_x_after, train_y_after = simplifyValues(train_values_X, train_values_y)
train_y_after[train_y_after == 10] = 0
sizeTrain = train_x_after.shape[0] - 1
test_x_after, test_y_after = simplifyValues(test_values_X, test_values_y)
sizeTest = test_x_after.shape[0] - 1
test_y_after[test_y_after == 10] = 0
del train_values_X, train_values_y, test_values_y, test_values_X
'''
#test DMIN avec toutes les valeurs des données train et test
#classesDMIN = classifieurDMIN(train_x_after, train_y_after, sizeTrain)
#print("DMIN with ", sizeTest , " values, percentage failed : " , testDMIN(classesDMIN, test_x_after, test_y_after, sizeTest), "%")


#tests préprocessing d'Elodie
#sans pre procc
#classesDMIN = classifieurDMIN(train_x_after, train_y_after, sizeTrain)
#print("DMIN with ", sizeTest , " values, w/o preprocessing, percentage failed : " , testDMIN(classesDMIN, test_x_after, test_y_after, sizeTest), "%")

# websites
# https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/
# https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a?fbclid=IwAR1-Qj9WihMYmdxM5zRsKY-pR0ffplMNrUXG_MY4unn9-bc_1TuESEi6tY8
#print('flatten...')

#train_x_greyscale = rgb2gray(train_x_after).astype(np.float32)
#test_x_greyscale = rgb2gray(test_x_after).astype(np.float32)

#filteredTrain = applyFilter(train_x_greyscale, 73257)
#filteredTest = applyFilter(test_x_greyscale, 26032)
#train_x_flatten = flattenData(filteredTrain)
#test_x_flatten = flattenData(filteredTest)
'''
train_x_flatten = flattenData(train_x_after)
test_x_flatten = flattenData(test_x_after)

trainX = train_x_flatten[0:1000]
trainY = np.ravel(train_y_after[0:1000])
testX = test_x_flatten[0:1000]
testY = np.ravel(test_y_after[0:1000])
'''


#NEW
print('new load')
data_x, data_y = load_data_flatten('train_32x32.mat',1)
test_x, test_y = load_data_flatten('test_32x32.mat',1)
#print(data_x.shape)
#print(data_y.shape)
trainX = data_x[0:2000]
trainY = data_y[0:2000]
testX = test_x[0:2000]
testY = test_y[0:2000]
print('load finished')

#knn_system(trainX, trainY, testX, testY)
#svm_system(trainX, trainY, testX, testY)
dmin_system(trainX, trainY, testX, testY)

print('with acp')

trainX = acp_transformation(trainX)
testX = acp_transformation(testX)
#knn_system(trainX, trainY, testX, testY)
#svm_system(trainX, trainY, testX, testY)
dmin_system(trainX, trainY, testX, testY)


'''
print('...finished')
print("--- %s seconds ---" % (time.time() - start_time))
print('pca...')
#pca = PCA(n_components=20)
#train_x_neig = pca.fit_transform(train_x_svm)
#test_x_neig = pca.fit_transform(test_x_svm)
print('...finished')
print("--- %s seconds ---" % (time.time() - start_time))
print('knn...')
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
print('...finished')
print("--- %s seconds ---" % (time.time() - start_time))
# Fit the classifier to the data
print('fit...')
knn.fit(train_x_svm,train_y_svm)
print('...finished')
print("--- %s seconds ---" % (time.time() - start_time))
print('score...')
#print(knn.predict(test_x_neig)[0:5])
#print(knn.score(test_x_svm, test_y_svm))
print('...finished')
'''
#svm = svm_classifieur(train_x_svm, train_y_svm)
#accuracy, svm_predictions = svm_result(svm, test_x_svm, test_y_svm)

#confusion_matrix = svm_confusion_matrix(test_y_svm, svm_predictions)
#print('SVM : svc / linear, accuracy = ', accuracy)


#partie des preprocessing
'''
#only filter
filteredTrain = applyFilter(train_x_after, 73257)
filteredTest = applyFilter(test_x_after, 26032)

#avec pre procc
classesDMIN = classifieurDMIN(filteredTrain, train_y_after, sizeTrain)
print("DMIN with ", sizeTest , " values, w/ preprocessing #1, percentage failed : " , testDMIN(classesDMIN, filteredTest, test_y_after, sizeTest), "%")


#pre procc
train_x_greyscale = rgb2gray(train_x_after).astype(np.float32)
test_x_greyscale = rgb2gray(test_x_after).astype(np.float32)

filteredTrain = applyFilter(train_x_greyscale, 73257)
filteredTest = applyFilter(test_x_greyscale, 26032)

#avec pre procc
classesDMIN = classifieurDMIN(filteredTrain, train_y_after, sizeTrain)
print("DMIN with ", sizeTest , " values, w/ preprocessing #2, percentage failed : " , testDMIN(classesDMIN, filteredTest, test_y_after, sizeTest), "%")
'''

#displayAllImages(filteredTrain, train_y_after, 10, 10)
#displayAllImages(train_x_greyscale, train_y_after, 10, 10)
#
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

















