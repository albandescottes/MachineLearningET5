
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

#NEW
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

#NEW
#transformaiton des données avec l'acp, de base on transforme en 10 valeurs
def acp_transformation(data, pca_values=10):
	pca = PCA(n_components=pca_values)
	return pca.fit_transform(data)

#NEW
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

#NEW
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
		


def rgb2gray(images):
    #return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)
    return np.expand_dims(np.dot(images, [1, 1, 1]), axis=3)

def load_data(path):
	data = loadmat(path)
	return data['X'], data['y']

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

def applyFilter(img, size):
	imgFiltered = [] 
	for i in range (0, size):
		mean = threshold_mean(img[i])
		#mid = img[i][16][16]

		hyst = filters.apply_hysteresis_threshold(img[i]-5, mean, mean+5)

		#hyst = filters.apply_hysteresis_threshold(img[i], mid-5, mid+5)
		imgFiltered.append(hyst)
	return np.array(imgFiltered)

#NEW MAIN
start_time = time.time()

#NEW
time_normal = time.time()
#data_x, data_y = load_data_flatten('train_32x32.mat')
#test_x, test_y = load_data_flatten('test_32x32.mat')
time_normal = (time.time() - time_normal)

#data = load_data('test_32x32.mat')

X_test, y_test = load_data('test_32x32.mat')
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
data_grey = rgb2gray(X_test).astype(np.float32)
#filteredTest = applyFilter(data_grey, 26032)
displayAllImages(filteredTest, y_test, 1, 10)

#data_x = []
#data_y = []
#for i in range(0, data['X'].shape[3]):
#	if preprocessing == 0:
#		data_x.append(data['X'][:,:,:,i].flatten())
#	elif preprocessing == 1:
#		data_x.append(preprocessing_image(data['X'][:,:,:,i]).flatten())
#	elif preprocessing == 2:
#		data_x.append(np.expand_dims(np.dot(data['X'][:,:,:,i], [0.2990, 0.5870, 0.1140]), axis=3))
#	elif preprocessing == 3:
#		data_x.append(preprocessing_image(np.expand_dims(np.dot(data['X'][:,:,:,i], [0.2990, 0.5870, 0.1140]), axis=3)).flatten())


#trainAcc =[]
#testAcc = []
#execTime = []

#svm_system(data_x[0:2500], data_y[0:2500], test_x[0:2500], test_y[0:2500])


'''
#x = ['2k/26k', '5k/26k', '10k/26k', '20k/26k', '40k/26k', '40k/26k', '73k/26k']
trainX1 = data_x[0:30000]
trainY1 = data_y[0:30000]
testX1 = test_x[0:26000]
testY1 = test_y[0:26000]
train, test, tps = dmin_system(trainX1, trainY1, testX1, testY1)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + time_normal)
ttime = time.time()
trainX2 = acp_transformation(data_x[0:30000])
trainY2 = data_y[0:30000]
testX2 = acp_transformation(test_x[0:26000])
testY2 = test_y[0:26000]
train, test, tps = dmin_system(trainX2, trainY2, testX2, testY2)
ttime = (time.time() - ttime)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + ttime + time_normal)

time_pre = time.time()
data_x, data_y = load_data_flatten('train_32x32.mat',3)
test_x, test_y = load_data_flatten('test_32x32.mat',3)
time_pre = (time.time() - time_pre)

ttime = time.time()
trainX3 = acp_transformation(data_x[0:30000])
trainY3 = data_y[0:30000]
testX3 = acp_transformation(test_x[0:26000])
testY3 = test_y[0:26000]
train, test, tps = dmin_system(trainX3, trainY3, testX3, testY3)
ttime = (time.time() - ttime)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + ttime + time_pre)

trainX3 = data_x[0:30000]
trainY3 = data_y[0:30000]
testX3 = test_x[0:26000]
testY3 = test_y[0:26000]
train, test, tps = dmin_system(trainX3, trainY3, testX3, testY3)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + time_pre)

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
'''
'''
t = ['DMIN', 'ACP+DMIN', 'PRE+ACP+DMIN', 'PRE+DMIN']
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Configuration')
ax1.set_ylabel('précision', color=color)
line_train, = ax1.plot(t, trainAcc, 'ro', label='Train')
line_test, = ax1.plot(t, testAcc, 'rX', label='Test')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 
color = 'tab:green'
ax2.set_ylabel('temps d\'execution', color=color)
line_exec, = ax2.plot(t, execTime, 'go', label='Time')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Classifieur DMIN')
plt.legend(handles=[line_train, line_test, line_exec])
plt.show()
print(t)
print(trainAcc)
print(testAcc)
print(execTime)

plt.xlabel('Nombre de données')
plt.ylabel('précision')
plt.title('DMIN')
plt.ylim(0.05,0.2)
line_train, =plt.plot(x, trainAcc, 'ro', label="Train")
line_test, = plt.plot(x, testAcc, 'go', label="Test")
plt.legend(handles=[line_train, line_test])
plt.show()
'''

'''
data_x = data_x[0:10000]
data_y = data_y[0:10000]
test_x = test_x[0:10000]
test_y = test_y[0:10000]

print('1')
trainX1 = acp_transformation(data_x,100)
trainY1 = data_y
testX1 = acp_transformation(test_x, 100)
testY1 = test_y

print('2')
trainX2 = acp_transformation(data_x,150)
trainY2 = data_y
testX2 = acp_transformation(test_x,150)
testY2 = test_y
print('3')
trainX3 = acp_transformation(data_x,200)
trainY3 = data_y
testX3 = acp_transformation(test_x,200)
testY3 = test_y
print('4')
trainX4 = acp_transformation(data_x,250)
trainY4 = data_y
testX4 = acp_transformation(test_x,250)
testY4 = test_y
print('5')
trainX5 = acp_transformation(data_x,300)
trainY5 = data_y
testX5 = acp_transformation(test_x,300)
testY5 = test_y
print('6')
trainX6 = acp_transformation(data_x,500)
trainY6 = data_y
testX6 = acp_transformation(test_x,500)
testY6 = test_y
'''

'''
#SVM
x = ['SVM', 'ACP+SVM', 'PRE+ACP+SVM', 'PRE+SVM']
trainX1 = data_x[0:3000]
trainY1 = data_y[0:3000]
testX1 = test_x[0:3000]
testY1 = test_y[0:3000]
#trainX1 = trainX1[0:500]
#trainY1 = trainY1[0:500]
#testX1 = testX1[0:1000]
#testY1 = testY1[0:1000]

train, test, tps = knn_system(trainX1, trainY1, testX1, testY1, 10)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + time_normal)

ttime = time.time()
trainX2 = acp_transformation(data_x[0:3000], 20)
trainY2 = data_y[0:3000]
testX2 = acp_transformation(test_x[0:3000], 20)
testY2 = test_y[0:3000]
ttime = (time.time() - ttime)
#trainX2 = trainX2[0:500]
#trainY2 = trainY2[0:500]
#testX2 = testX2[0:1000]
#testY2 = testY2[0:1000]
train, test, tps = knn_system(trainX2, trainY2, testX2, testY2, 10)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + time_normal + ttime)

time_pre = time.time()
data_x, data_y = load_data_flatten('train_32x32.mat',3)
test_x, test_y = load_data_flatten('test_32x32.mat',3)
time_pre = (time.time() - time_pre)

ttime = time.time()
trainX3 = acp_transformation(data_x[0:3000], 20)
trainY3 = data_y[0:3000]
testX3 = acp_transformation(test_x[0:2000], 20)
testY3 = test_y[0:3000]
ttime = (time.time() - ttime)
#trainX3 = trainX3[0:500]
#trainY3 = trainY3[0:500]
#testX3 = testX3[0:1000]
#testY3 = testY3[0:1000]
train, test, tps = knn_system(trainX3, trainY3, testX3, testY3, 10)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + time_pre + ttime)


trainX4 = data_x[0:3000]
trainY4 = data_y[0:3000]
testX4 = test_x[0:3000]
testY4 = test_y[0:3000]
#trainX4 = trainX4[0:500]
#trainY4 = trainY4[0:500]
#testX4 = testX4[0:1000]
#testY4 = testY4[0:1000]
train, test, tps = knn_system(trainX4, trainY4, testX4, testY4, 10)
trainAcc.append(train)
testAcc.append(test)
execTime.append(tps + time_pre)

t = ['KNN', 'ACP+KNN', 'PRE+ACP+KNN', 'PRE+KNN']
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Configuration')
ax1.set_ylabel('précision', color=color)
line_train, = ax1.plot(t, trainAcc, 'ro', label='Train')
line_test, = ax1.plot(t, testAcc, 'rX', label='Test')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 
color = 'tab:green'
ax2.set_ylabel('temps d\'execution', color=color)
line_exec, = ax2.plot(t, execTime, 'go', label='Time')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Classifieur KNN')
plt.legend(handles=[line_train, line_test, line_exec])
plt.show()
print(t)
print(trainAcc)
print(testAcc)
print(execTime)
'''
'''
#trainX5 = data_x[0:500]
#trainY5 = data_y[0:500]
#testX5 = test_x[0:1000]
#testY5 = test_y[0:1000]
trainX5 = trainX5[0:500]
trainY5 = trainY5[0:500]
testX5 = testX5[0:1000]
testY5 = testY5[0:1000]
train, test, tps = svm_system(trainX5, trainY5, testX5, testY5, 'poly')
trainAcc.append(train)
testAcc.append(test)
#trainX6 = data_x[0:500]
#trainY6 = data_y[0:500]
#testX6 = test_x[0:1000]
#testY6 = test_y[0:1000]
trainX6 = trainX6[0:500]
trainY6 = trainY6[0:500]
testX6 = testX6[0:1000]
testY6 = testY6[0:1000]
train, test, tps = svm_system(trainX6, trainY6, testX6, testY6, 'poly')
trainAcc.append(train)
testAcc.append(test)

plt.xlabel('kernel')
plt.ylabel('précision')
plt.title('Preprocessing + SVM 500 train / 1000 test')
plt.ylim(0.05,1)
line_train, =plt.plot(x, trainAcc, 'ro', label="Train")
line_test, = plt.plot(x, testAcc, 'go', label="Test")
plt.legend(handles=[line_train, line_test])
plt.show()
'''
'''
print('1')
trainX1 = acp_transformation(data_x,10)
trainY1 = data_y
testX1 = acp_transformation(test_x, 10)
testY1 = test_y

print('2')
trainX2 = acp_transformation(data_x,20)
trainY2 = data_y
testX2 = acp_transformation(test_x,20)
testY2 = test_y
print('3')
trainX3 = acp_transformation(data_x,30)
trainY3 = data_y
testX3 = acp_transformation(test_x,30)
testY3 = test_y
print('4')
trainX4 = acp_transformation(data_x,40)
trainY4 = data_y
testX4 = acp_transformation(test_x,40)
testY4 = test_y
print('5')
trainX5 = acp_transformation(data_x,50)
trainY5 = data_y
testX5 = acp_transformation(test_x,50)
testY5 = test_y
print('6')
trainX6 = acp_transformation(data_x,60)
trainY6 = data_y
testX6 = acp_transformation(test_x,60)
testY6 = test_y
'''
'''
x = ['k=1', 'k=3', 'k=5', 'k=7', 'k=9', 'k=11', 'k=13']
trainX1 = data_x[0:3000]
trainY1 = data_y[0:3000]
testX1 = test_x[0:3000]
testY1 = test_y[0:3000]
#trainX1 = trainX1[0:3000]
#trainY1 = trainY1[0:3000]
#testX1 = testX1[0:3000]
#testY1 = testY1[0:3000]
train, test, tps = knn_system(trainX1, trainY1, testX1, testY1, 1)
trainAcc.append(train)
testAcc.append(test)
trainX2 = data_x[0:3000]
trainY2 = data_y[0:3000]
testX2 = test_x[0:3000]
testY2 = test_y[0:3000]
#trainX2 = trainX2[0:3000]
#trainY2 = trainY2[0:3000]
#testX2 = testX2[0:3000]
#testY2 = testY2[0:3000]
train, test, tps = knn_system(trainX2, trainY2, testX2, testY2, 3)
trainAcc.append(train)
testAcc.append(test)
trainX3 = data_x[0:3000]
trainY3 = data_y[0:3000]
testX3 = test_x[0:3000]
testY3 = test_y[0:3000]
#trainX3 = trainX3[0:3000]
#trainY3 = trainY3[0:3000]
#testX3 = testX3[0:3000]
#testY3 = testY3[0:3000]
train, test, tps = knn_system(trainX3, trainY3, testX3, testY3, 5)
trainAcc.append(train)
testAcc.append(test)
trainX4 = data_x[0:3000]
trainY4 = data_y[0:3000]
testX4 = test_x[0:3000]
testY4 = test_y[0:3000]
#trainX4 = trainX4[0:3000]
#trainY4 = trainY4[0:3000]
#testX4 = testX4[0:3000]
#testY4 = testY4[0:3000]
train, test, tps = knn_system(trainX4, trainY4, testX4, testY4, 7)
trainAcc.append(train)
testAcc.append(test)
trainX5 = data_x[0:3000]
trainY5 = data_y[0:3000]
testX5 = test_x[0:3000]
testY5 = test_y[0:3000]
#trainX5 = trainX5[0:3000]
#trainY5 = trainY5[0:3000]
#testX5 = testX5[0:3000]
#testY5 = testY5[0:3000]
train, test, tps = knn_system(trainX5, trainY5, testX5, testY5, 9)
trainAcc.append(train)
testAcc.append(test)
trainX6 = data_x[0:3000]
trainY6 = data_y[0:3000]
testX6 = test_x[0:3000]
testY6 = test_y[0:3000]
#trainX6 = trainX6[0:3000]
#trainY6 = trainY6[0:3000]
#testX6 = testX6[0:3000]
#testY6 = testY6[0:3000]
train, test, tps = knn_system(trainX6, trainY6, testX6, testY6, 11)
trainAcc.append(train)
testAcc.append(test)

trainX7 = data_x[0:3000]
trainY7 = data_y[0:3000]
testX7 = test_x[0:3000]
testY7 = test_y[0:3000]
train, test, tps = knn_system(trainX7, trainY7, testX7, testY7, 13)
trainAcc.append(train)
testAcc.append(test)

plt.xlabel('k')
plt.ylabel('précision')
plt.title('preprocessing + KNN 3000 train / 3000 test')
plt.ylim(0.3,0.7)
line_train, =plt.plot(x, trainAcc, 'ro', label="Train")
line_test, = plt.plot(x, testAcc, 'go', label="Test")
plt.legend(handles=[line_train, line_test])
plt.show()
'''
print("-total time of execution - %s seconds ---" % (time.time() - start_time))
print("End of program")

