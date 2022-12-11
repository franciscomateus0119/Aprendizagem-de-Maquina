import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from matplotlib.colors import ListedColormap
from collections import Counter
from models import KMeans
import math
from tqdm.notebook import trange, tqdm


def mean_and_std(array):
    mean = np.mean(array)
    std = np.sqrt(np.mean((array - mean) ** 2))
    return mean,std
def standard_deviation(array):
    return np.sqrt(np.mean((array - np.mean(array)) ** 2))
def calc_acc(y_true,y_pred):
    return np.sum(np.equal(y_true, y_pred)) / len(y_true)
def calc_acc2(y_true,y_pred):
    acertos = 0
    for y1,y2 in zip(y_true,y_pred):
      if y1 == y2:
        acertos+=1        
    return acertos/ len(y_true)
def get_closest(value, lista):
    array = np.array(lista)
    index = np.argmin(abs(array - value))
    return index
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
def create_cfmat(y_true,y_pred):
    #Identify labels and nº of labels
    labels = np.unique(y_true)
    size = len(labels)
    #Create a zeroed square matrix of size n, with n = number of different classes 
    matrix = pd.DataFrame(np.zeros((size, size),dtype=int),index = labels, columns = labels)
    #For each possible combination of two classes, add 1 for every prediction
    for true_class,pred_class in zip(y_true,y_pred):
      matrix.loc[true_class, pred_class] += 1
    return matrix.values
def make_plot(classifier,title,data,data_labels):
    step = .02
    cmap_base = ListedColormap(['darkseagreen', 'cornflowerblue', 'slategrey'])
    cmap_labels = ['white', 'yellow', 'orange']


    #Find Minimum and Maximum value for the data
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    #Create Grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),np.arange(y_min, y_max, step))

    #Obtain predicted values for each of the grid's points
    Z = np.array(classifier.predict(np.c_[xx.ravel(), yy.ravel()]))
    for i in range(len(np.unique(Z))):
      Z[np.where(Z == np.unique(Z)[i])] = i
    Z = Z.reshape(xx.shape)

    #Make Contours
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_base)

    #Create a Scatterplot with the data
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=data_labels,
    palette=cmap_labels, alpha=1.0, edgecolor="black")

    #Set the limits for the variables
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title(title)

    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()

def get_best_K(X,y):
    Kmax = int(len(X)*0.2)
    if Kmax > 50 and Kmax > len(np.unique(y)):
      Kmax = 50
    elif Kmax > 50 and Kmax <= len(np.unique(y)):
      Kmax = len(np.unique(y))
    accs = np.array([])
    for i in trange(20):
      #print(f'Iteration {i}')
      accuracies = np.asarray([])
      for K in range(1,Kmax):
        #Separating train and test subsets from original dataset, with 80/20 proportions
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        #Normalization
        u, s = mean_and_std(X_train)
        X_train = (X_train - u) / s
        X_test = (X_test - u) / s

        #Training
        kmeans = KMeans(K)
        kmeans.train(X_train,y_train)
        predictions = kmeans.predict(X_test)
        accuracy = calc_acc(y_test, predictions)
        accuracies = np.append(accuracies,accuracy)
      if len(accs) == 0:
        accs = accuracies
      else:
        accs = np.vstack((accs,accuracies))
    means = np.asarray([])
    for i in range(accs.shape[1]):
      print(f'K={i+1} = {np.mean(accs[:,i])}')
      means = np.append(means,np.mean(accs[:,i]))
    max_K = np.argmax(means)+1
    print(f'Found Highest Accuracy {np.max(means)} with K={max_K}')
    return max_K

def plot_confusion_matrix(confusion_matrix, title, hitrate, std, labels):
  ax = sns.heatmap(confusion_matrix, annot=True,cmap='Blues',fmt="d",annot_kws={'size': 25})
  ax.set_title(f'{title}\nAccuracy {hitrate}, STD = {std}');
  ax.set_xlabel('\nPredicted')
  ax.set_ylabel('\nExpected');
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()