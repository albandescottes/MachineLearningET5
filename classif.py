import sys
import os.path
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import time
import numpy as np
from skimage.filters import threshold_mean
from skimage import data, filters


def load_data_flatten(path, preprocessing=0):
	start_time = time.time()
	print('--chargement de {}'.format(path))
	data = loadmat(path)
	data_x = []
	data_y = []
	for i in range(0, data['X'].shape[3]):
		data_x.append(preprocessing_image(np.expand_dims(np.dot(data['X'][:,:,:,i], [0.2990, 0.5870, 0.1140]), axis=3)).flatten())
		data_y.append(data['y'][i])
	data_y = np.array(data_y)
	data_y[data_y == 10] = 0
	print("--temps d\'exécution %s secondess" % (time.time() - start_time))
	print('--')
	return np.array(data_x), data_y

def preprocessing_image(img):
	mean = threshold_mean(img)
	return filters.apply_hysteresis_threshold(img -5, mean, mean + 5)

def knn_system(trainX, trainY, testX, testY):
	start_time = time.time()
	print('--classifieur KNN')
	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(trainX,trainY)
	print('--fichier d\'entrainement')
	knn_prediction = knn.predict(trainX)
	accuracy_train, confusion_matrix = results_system(knn_prediction, trainY)
	print('--matrice de confusion')
	print('--fichier d\'entrainement : précison = ', accuracy_train)
	print_c_m(confusion_matrix)
	print('--')
	print('--fichier de test')
	knn_prediction = knn.predict(testX)
	accuracy_test, confusion_matrix = results_system(knn_prediction, testY)
	print('--fichier test - précision = ', accuracy_test)
	print('--matrice de confusion')
	print_c_m(confusion_matrix)
	print("--temps d\'exécution %s secondess" % (time.time() - start_time))
	print('--')

def results_system(pred, val):
	cm = np.zeros((10,10))
	for i in range(0, pred.shape[0]):
		cm[val[i], pred[i]] += 1
	acc = np.trace(cm) / pred.shape[0]
	return acc, cm

def print_c_m(m):
	print('\t| 0\t| 1\t| 2\t| 3\t| 4\t| 5\t| 6\t| 7\t| 8\t| 9')
	for i in range(0,10):
		print('-------------------------------------------------------------------------------------')
		print('{}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}\t| {}'.format(i, m[i,0].astype(np.int32), \
			m[i,1].astype(np.int32), m[i,2].astype(np.int32), m[i,3].astype(np.int32), m[i,4].astype(np.int32),\
			 m[i,5].astype(np.int32), m[i,6].astype(np.int32), m[i,7].astype(np.int32), m[i,8].astype(np.int32), \
			 m[i,9].astype(np.int32)))

if __name__ == "__main__":
	if(sys.argv[1] != '--data' or len(sys.argv) != 3):
		print('mauvais passage d\'argument')
		print('#python3 classif.py --data nom_fichier.mat')
	elif(not os.path.isfile(sys.argv[2])):
		print(sys.argv[2], ' n\'existe pas')
	elif(not os.path.isfile('train_32x32.mat')):
		print('le ficher train_32x32.mat est manquant')
	else:
		trainX,trainY = load_data_flatten('train_32x32.mat')
		testX,testY = load_data_flatten(sys.argv[2])
		knn_system(trainX[0:15000], trainY[0:15000], testX, testY)