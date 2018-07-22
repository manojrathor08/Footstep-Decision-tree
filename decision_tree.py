#import numpy as np
#
## Here is a list of five 10x10 arrays:
#x=[np.random.random((10,10)) for _ in range(5)]
#
#y=np.dstack(x)
#print(y.shape)
## (10, 10, 5)
#
## To get the shape to be Nx10x10, you could  use rollaxis:
#y=np.rollaxis(y,-1)
#print(y.shape)
## (5, 10, 10)
#import os
#from numpy import genfromtxt
#path = '/home/manoj/Desktop/footstep_recognition/data'
#datadir = os.listdir(path)
#for folder in datadir:
#    print(folder)
#    for files in os.listdir(os.path.join(path,folder)):
#        data = genfromtxt(os.path.join(path,folder,files),delimiter = ',')
#        if folder=='ak':
#            #print(files)
#            #print(data.shape)
#            if data.shape[0]!=60:
#                print("FILES",files)
#            if data.shape[1]!=18:
#                print("FILES",files)
# importing necessary libraries
# importing necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from data import DataSet
import time
batch_size = 32
hidden_size = 10
use_dropout=True
vocabulary = 6
data_type = 'features'
seq_length = 60
class_limit =  6
image_shape = None
data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )
# generator = data.frame_generator(batch_size, 'train', data_type)
# # for f in generator:
# #     print(f)
# val_generator = data.frame_generator(batch_size, 'test', data_type)
X_tr, y_tr = data.get_all_sequences_in_memory('train', data_type)
X_train= X_tr.reshape(780,1080)
y_train = np.zeros(780)
j = 0
for i in y_tr:
    #print(np.argmax(i))
    y_train[j] = np.argmax(i)
    j +=1
#print(X_train.shape)
#print(y_train.shape)
X_te, y_te = data.get_all_sequences_in_memory('test', data_type)
X_test = X_te.reshape(192,1080)
y_test = np.zeros(192)
j = 0
for i in y_te:
    #print(np.argmax(i))
    y_test[j] = np.argmax(i)
    j +=1
#print(X_test.shape)
#print(y_test.shape)

# loading the iris dataset
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
# dividing X, y into train and test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 9000000).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
np.set_printoptions(precision=2)
#class_names = iris.target_names
class_names = [1,2,3,4,5,6]
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()   