
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import time

from skimage.filters import threshold_mean
from skimage import data, filters

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

#NEW
#creation d'une classe pour uniformiser l'utilisation de ce classifieur
class DMIN:
	def __init__(self):
		self.predictions = []
		self.finalClasses = []

	def fit(self, dataX, dataY):
		del self.finalClasses[:]
		classes = []
		for i in range(0,10):
			classes.append([])
		for i in range(0 ,dataY.shape[0]):
			classes[dataY[i]].append(dataX[i])
		for i in range(0,10):
			self.finalClasses.append(np.mean(classes[i], axis=0))

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

#NEW
#flatten and cut in x and y
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

#NEW
#méthode SVM, utilisation de SVC
def svm_system(trainX, trainY, testX, testY, ker='linear', deg='3'):
	start_time = time.time()
	print('----SVM--------------')
	svc = SVC(kernel = ker, degree=deg)
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

#NEW
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


#NEW MAIN
start_time = time.time()

# websites
# https://www.geeksforgeeks.org/multiclass-classification-using-scikit-learn/
# https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a?fbclid=IwAR1-Qj9WihMYmdxM5zRsKY-pR0ffplMNrUXG_MY4unn9-bc_1TuESEi6tY8

#NEW
data_x, data_y = load_data_flatten('train_32x32.mat')
test_x, test_y = load_data_flatten('test_32x32.mat')
trainAcc =[]
testAcc = []

'''
x = ['2k/26k', '5k/26k', '10k/26k', '20k/26k', '40k/26k', '60k/26k', '73k/26k']
#print(data_x.shape)
#print(data_y.shape)
trainX1 = data_x[0:2000]
trainY1 = data_y[0:2000]
testX1 = test_x[0:26000]
testY1 = test_y[0:26000]
train, test, tps = dmin_system(trainX1, trainY1, testX1, testY1)
trainAcc.append(train)
testAcc.append(test)
trainX2 = data_x[0:5000]
trainY2 = data_y[0:5000]
testX2 = test_x[0:26000]
testY2 = test_y[0:26000]
train, test, tps = dmin_system(trainX2, trainY2, testX2, testY2)
trainAcc.append(train)
testAcc.append(test)
trainX3 = data_x[0:10000]
trainY3 = data_y[0:10000]
testX3 = test_x[0:26000]
testY3 = test_y[0:26000]
train, test, tps = dmin_system(trainX3, trainY3, testX3, testY3)
trainAcc.append(train)
testAcc.append(test)
trainX4 = data_x[0:20000]
trainY4 = data_y[0:20000]
testX4 = test_x[0:26000]
testY4 = test_y[0:26000]
train, test, tps = dmin_system(trainX4, trainY4, testX4, testY4)
trainAcc.append(train)
testAcc.append(test)
trainX5 = data_x[0:40000]
trainY5 = data_y[0:40000]
testX5 = test_x[0:26000]
testY5 = test_y[0:26000]
train, test, tps = dmin_system(trainX5, trainY5, testX5, testY5)
trainAcc.append(train)
testAcc.append(test)
trainX6 = data_x[0:60000]
trainY6 = data_y[0:60000]
testX6 = test_x[0:26000]
testY6 = test_y[0:26000]
train, test, tps = dmin_system(trainX6, trainY6, testX6, testY6)
trainAcc.append(train)
testAcc.append(test)
trainX7 = data_x[0:73000]
trainY7 = data_y[0:73000]
testX7 = test_x[0:26000]
testY7 = test_y[0:26000]
train, test, tps = dmin_system(trainX7, trainY7, testX7, testY7)
trainAcc.append(train)
testAcc.append(test)

plt.xlabel('nombre train / nombre test')
plt.ylabel('précision')
plt.title('DMIN')
plt.ylim(0.05,0.2)
line_train, =plt.plot(x, trainAcc, 'ro', label="Train")
line_test, = plt.plot(x, testAcc, 'go', label="Test")
plt.legend(handles=[line_train, line_test])
plt.show()
'''


#SVM
x = ['2', '3', '4', '5']
trainX1 = data_x[0:500]
trainY1 = data_y[0:500]
testX1 = test_x[0:1000]
testY1 = test_y[0:1000]
train, test, tps = svm_system(trainX1, trainY1, testX1, testY1, 'poly', 2)
trainAcc.append(train)
testAcc.append(test)
trainX2 = data_x[0:500]
trainY2 = data_y[0:500]
testX2 = test_x[0:1000]
testY2 = test_y[0:1000]
train, test, tps = svm_system(trainX2, trainY2, testX2, testY2, 'poly', 3)
trainAcc.append(train)
testAcc.append(test)
trainX3 = data_x[0:500]
trainY3 = data_y[0:500]
testX3 = test_x[0:1000]
testY3 = test_y[0:1000]
train, test, tps = svm_system(trainX3, trainY3, testX3, testY3, 'poly', 4)
trainAcc.append(train)
testAcc.append(test)

trainX4 = data_x[0:500]
trainY4 = data_y[0:500]
testX4 = test_x[0:1000]
testY4 = test_y[0:1000]
train, test, tps = svm_system(trainX4, trainY4, testX4, testY4, 'poly', 5)
trainAcc.append(train)
testAcc.append(test)
'''
trainX5 = data_x[0:1000]
trainY5 = data_y[0:1000]
testX5 = test_x[0:1000]
testY5 = test_y[0:1000]
train, test, tps = svm_system(trainX5, trainY5, testX5, testY5)
trainAcc.append(train)
testAcc.append(test)
trainX6 = data_x[0:1500]
trainY6 = data_y[0:1500]
testX6 = test_x[0:1000]
testY6 = test_y[0:1000]
train, test, tps = svm_system(trainX6, trainY6, testX6, testY6)
trainAcc.append(train)
testAcc.append(test)
'''
plt.xlabel('kernel=\'poly\' degree')
plt.ylabel('précision')
plt.title('SVM')
plt.ylim(0.05,1)
line_train, =plt.plot(x, trainAcc, 'ro', label="Train")
line_test, = plt.plot(x, testAcc, 'go', label="Test")
plt.legend(handles=[line_train, line_test])
plt.show()


'''
x = ['k=3', 'k=4', 'k=5', 'k=6', 'k=7', 'k=8', 'k=9']
trainX1 = data_x[0:3000]
trainY1 = data_y[0:3000]
testX1 = test_x[0:3000]
testY1 = test_y[0:3000]
train, test, tps = knn_system(trainX1, trainY1, testX1, testY1, 3)
trainAcc.append(train)
testAcc.append(test)
trainX2 = data_x[0:3000]
trainY2 = data_y[0:3000]
testX2 = test_x[0:3000]
testY2 = test_y[0:3000]
train, test, tps = knn_system(trainX2, trainY2, testX2, testY2, 4)
trainAcc.append(train)
testAcc.append(test)
trainX3 = data_x[0:3000]
trainY3 = data_y[0:3000]
testX3 = test_x[0:3000]
testY3 = test_y[0:3000]
train, test, tps = knn_system(trainX3, trainY3, testX3, testY3, 5)
trainAcc.append(train)
testAcc.append(test)
trainX4 = data_x[0:3000]
trainY4 = data_y[0:3000]
testX4 = test_x[0:3000]
testY4 = test_y[0:3000]
train, test, tps = knn_system(trainX4, trainY4, testX4, testY4, 6)
trainAcc.append(train)
testAcc.append(test)
trainX5 = data_x[0:3000]
trainY5 = data_y[0:3000]
testX5 = test_x[0:3000]
testY5 = test_y[0:3000]
train, test, tps = knn_system(trainX5, trainY5, testX5, testY5, 7)
trainAcc.append(train)
testAcc.append(test)
trainX6 = data_x[0:3000]
trainY6 = data_y[0:3000]
testX6 = test_x[0:3000]
testY6 = test_y[0:3000]
train, test, tps = knn_system(trainX6, trainY6, testX6, testY6, 8)
trainAcc.append(train)
testAcc.append(test)
trainX7 = data_x[0:3000]
trainY7 = data_y[0:3000]
testX7 = test_x[0:3000]
testY7 = test_y[0:3000]
train, test, tps = knn_system(trainX7, trainY7, testX7, testY7, 9)
trainAcc.append(train)
testAcc.append(test)

plt.xlabel('k')
plt.ylabel('précision')
plt.title('KNN')
plt.ylim(0.2,0.6)
line_train, =plt.plot(x, trainAcc, 'ro', label="Train")
line_test, = plt.plot(x, testAcc, 'go', label="Test")
plt.legend(handles=[line_train, line_test])
plt.show()
'''
print("-total time of execution - %s seconds ---" % (time.time() - start_time))
print("End of program")

