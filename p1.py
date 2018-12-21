# Alban Descottes & Elodie Lam
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from skimage.filters import threshold_mean
from skimage import data, filters

import numpy as np
import matplotlib.pyplot as plt
import time

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

#creation d'une classe pour uniformiser l'utilisation de ce classifieur
class DMIN:
	def __init__(self):
		self.predictions = []
		self.finalClasses = []

	#méthode pour créer le classifieur dmin
	def fit(self, dataX, dataY):
		del self.finalClasses[:]
		classes = []
		for i in range(0,10):
			classes.append([])
		for i in range(0 ,dataY.shape[0]):
			classes[dataY[i]].append(dataX[i])
		for i in range(0,10):
			self.finalClasses.append(np.mean(classes[i], axis=0))

	#méhtode qui retourne les prédicitons sous forme d'une array
	def predict(self, dataX):
		del self.predictions[:]
		for i in range(0, len(dataX)):
			bestValue = np.linalg.norm(self.finalClasses[0]-dataX[i])
			bestMatch = 0
			for c in range(1, 10):
				temp = np.linalg.norm(self.finalClasses[c]-dataX[i])
				if(temp < bestValue):
					bestValue = temp
					bestMatch = c
			self.predictions.append(bestMatch)
		return np.array(self.predictions)

#méthode qui récupère les données et les transforme en données pour les différents classifieurs
def load_data_flatten(path, preprocessing=0):
	start_time = time.time()
	print('----LOAD-OF-{}---'.format(path))
	data = loadmat(path)
	data_x = []
	data_y = []
	for i in range(0, data['X'].shape[3]):
		if preprocessing == 0:
			data_x.append(data['X'][:,:,:,i].flatten())
		elif preprocessing == 1:
			data_x.append(preprocessing_image(data['X'][:,:,:,i]).flatten())
		elif preprocessing == 2:
			data_x.append(np.expand_dims(np.dot(data['X'][:,:,:,i], [0.2990, 0.5870, 0.1140]), axis=3))
		elif preprocessing == 3:
			data_x.append(preprocessing_image(np.expand_dims(np.dot(data['X'][:,:,:,i], [0.2990, 0.5870, 0.1140]), axis=3)).flatten())

		data_y.append(data['y'][i])
	data_y[data_y == 10] = 0
	data_y = np.array(data_y)
	data_y[data_y == 10] = 0
	print("execution time for loading %s seconds ---" % (time.time() - start_time))
	print('---------------------')
	return np.array(data_x), data_y

#méthode pour l'application du preprocessing sur une image
def preprocessing_image(img):
	mean = threshold_mean(img)
	return filters.apply_hysteresis_threshold(img -5, mean, mean + 5)

#transformaiton des données avec l'acp, de base on transforme en 10 valeurs
def acp_transformation(data, pca_values=10):
	pca = PCA(n_components=pca_values)
	return pca.fit_transform(data)

#méthode pour le classifieur DMIN
def dmin_system(trainX, trainY, testX, testY):
	start_time = time.time()
	print('----DMIN-------------')
	dmin = DMIN()
	dmin.fit(trainX, trainY)
	print('----TRAIN------------')
	dmin_predictions = dmin.predict(trainX)
	accuracy_train, confusion_matrix = results_system(dmin_predictions, trainY)
	print('dmin accuracy = ', accuracy_train)
	print_confusion_matrix(confusion_matrix)
	print('---------------------')
	print('----TEST-------------')
	dmin_predictions = dmin.predict(testX)
	accuracy_test, confusion_matrix = results_system(dmin_predictions, testY)
	print('dmin accuracy = ', accuracy_test)
	print_confusion_matrix(confusion_matrix)
	print("execution time for DMIN %s seconds ---" % (time.time() - start_time))
	print('---------------------')
	return accuracy_train, accuracy_test, (time.time() - start_time)

#méthode SVM, utilisation de SVC
def svm_system(trainX, trainY, testX, testY, ker='poly'):
	start_time = time.time()
	print('----SVM--------------')
	svc = SVC(kernel = ker)
	svc.fit(trainX, trainY)
	print('----TRAIN------------')
	svc_predictions = svc.predict(trainX)
	accuracy_train, confusion_matrix = results_system(svc_predictions, trainY)
	print('svc accuracy = ', accuracy_train)
	print_confusion_matrix(confusion_matrix)
	print('---------------------')
	print('----TEST-------------')
	svc_predictions = svc.predict(testX)
	accuracy_test, confusion_matrix = results_system(svc_predictions, testY)
	print('svc accuracy = ', accuracy_test)
	print_confusion_matrix(confusion_matrix)
	print("execution time for svm %s seconds ---" % (time.time() - start_time))
	print('---------------------')
	return accuracy_train, accuracy_test, (time.time() - start_time)

#méthode Neighbors, utilisation de KNeighborsClassifier
def knn_system(trainX, trainY, testX, testY, k=10):
	start_time = time.time()
	print('----KNN--------------')
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(trainX,trainY)
	print('----TRAIN------------')
	knn_prediction = knn.predict(trainX)
	accuracy_train, confusion_matrix = results_system(knn_prediction, trainY)
	print('knn accuracy = ', accuracy_train)
	print_confusion_matrix(confusion_matrix)
	print('---------------------')
	print('----TEST-------------')
	knn_prediction = knn.predict(testX)
	accuracy_test, confusion_matrix = results_system(knn_prediction, testY)
	print('knn accuracy = ', accuracy_test)
	print_confusion_matrix(confusion_matrix)
	print("execution time for knn %s seconds ---" % (time.time() - start_time))
	print('---------------------')
	return accuracy_train, accuracy_test, (time.time() - start_time)

#NEW
#méthode qui calcule la précision du système ainsi que la matrice de confusion
def results_system(predictions, values):
	confusion_matrix = np.zeros((10,10))
	for i in range(0, predictions.shape[0]):
		confusion_matrix[values[i], predictions[i]] += 1
	#print(confusion_matrix)
	accuracy = np.trace(confusion_matrix) / predictions.shape[0]
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