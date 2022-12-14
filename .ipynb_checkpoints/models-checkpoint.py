import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import math


class KNearestNeighbours:
  def __init__(self,k=5,metric="euclidean"):
    self.k = k
    self.metric = metric
    self.X = None
    self.y = None
  #Calculates the Euclidean Distance between Points
  def euclidean_distance(self,a,b):
    return np.sqrt((a-b).T @ (a-b))
  
  #Find the neighbours through the distances
  def get_neighbours(self,distances):
    if type(distances)!= 'numpy.ndarray':
      distances = np.array(distances)
    indexes = np.argsort(distances)
    distances = distances[indexes]
    neighbours = self.X[indexes][0:self.k]
    neighbours_labels = self.y[indexes][0:self.k]
    return distances,neighbours,neighbours_labels

  #Find the most common class and use it as the label for the sample
  def check_most_common(self, neighbours, neighbours_labels):
    counter_list = Counter(neighbours_labels).most_common()
    max_n_frequencies = counter_list[0][1]
    if len(counter_list) > 1:
      if counter_list[0][1] == counter_list[1][1]:
        c_idx = []
        for i in range(len(counter_list)):
          if counter_list[0][1] == counter_list[i][1]:
            c_idx.append(i)
        eligible_labels = np.array(counter_list)[c_idx]
        label_check = False
        i = 0
        while label_check == False:
          if neighbours_labels[i] in eligible_labels:
            label_check = True
          else:
            i += 1
        return neighbours_labels[i]
      else:
        return counter_list[0][0]
    else:
      return counter_list[0][0]
    
  #Train
  def train(self,X=None,y=None):
    #Shuffle and save
    if X is None:
      raise ValueError("Expected at least one training sample. None was given.")
    if y is None:
      raise ValueError("Expected at least one training label. None was given.")
    
    self.X = X 
    self.y = y
    
  #Predict from Samples
  def predict(self,samples):
    if samples.ndim == 1:
      if self.metric=="euclidean":
        distances = []
        for x in self.X:
          distances.append(self.euclidean_distance(x,samples))

      distances,neighbours,neighbours_labels = self.get_neighbours(distances)
      prediction = self.check_most_common(neighbours,neighbours_labels)
      return prediction

    elif samples.ndim ==2:
      if self.metric=="euclidean":
        predictions = []
        for sample in samples:
          distances = []
          for x in self.X:
            distances.append(self.euclidean_distance(x,sample))

          distances,neighbours,neighbours_labels = self.get_neighbours(distances)
          prediction = self.check_most_common(neighbours,neighbours_labels)
          predictions.append(prediction)
      return predictions
    else:
      raise ValueError("Expected ndim == 1 or ndim == 2.")

class DMC:
  def __init__(self,metric="euclidean"):
    self.metric = metric
    self.centroids = None
    self.labels = None
  #Calculates the Euclidean Distance between Points
  def euclidean_distance(self,a,b):
    return np.sqrt((a-b).T @ (a-b))
  
  #Find the neighbours and members of a cluster through the distances to the centroids
  def get_nearest_centroid(self,distances):
    if type(distances)!= 'numpy.ndarray':
      distances = np.array(distances)
    #Get the indexes of each distance, sorted from the smallest distance to the highest
    indexes = np.argsort(distances)
    distances = distances[indexes]

    #Get the centroid least distant to the given sample
    centroid = self.centroids[indexes[0]]
    centroid_label = self.labels[indexes[0]]
    return distances, centroid, centroid_label
  
  #Build Centroids
  def build_centroids(self,X,y):
    centroids = []
    #Do the sum of all patterns with same label
    for i in range(len(self.labels)):
      soma = []
      for x in (X[np.where(y == self.labels[i])]):
        soma.append(x)  

      #and store their mean 
      centroids.append(sum(soma)/len(np.where(y == self.labels[i])[0]))
    return centroids

  #Train
  def train(self,X=None,y=None):
    
    if X is None:
      raise ValueError("Expected at least one training sample. None was given.")
    if y is None:
      raise ValueError("Expected at least one training label. None was given.")
    sX = X
    sy = y
    self.labels = np.unique(sy)
    self.centroids = self.build_centroids(sX,sy)

  #Predict from Samples
  def predict(self,samples):
    if samples.ndim == 1:
      if self.metric == "euclidean":
        predictions = []
        distances = []
        for centroid in self.centroids:
          distances.append(self.euclidean_distance(centroid,samples))          
        distances, nearest_centroid, nearest_label = self.get_nearest_centroid(distances)
        predictions.append(nearest_label)
        return predictions
      else:
        raise ValueError("Expected valid metric type for calculating distances. Received: ",self.metric)
      
    elif samples.ndim == 2:
      if self.metric == "euclidean":
        predictions = []
        for sample in samples:
          distances = []
          for centroid in self.centroids:
            distances.append(self.euclidean_distance(centroid,sample))         
          distances, nearest_centroid, nearest_label = self.get_nearest_centroid(distances)
          predictions.append(nearest_label)
        return predictions
      else:
        raise ValueError("Expected valid metric type for calculating distances. Received: ",self.metric)
    else:
      raise ValueError("Expected ndim == 1 or ndim == 2. Received",samples.ndim)

class KMeans:
    '''
    A KMeans algorithm implementation for clustering and classification.
    '''
    def __init__(self, n_clusters, iterations=100,metric='euclidean'):
        '''
        self.k : total number of clusters and centroids
        self.iterations: how many iterations of updating the centroids should be run
        self.metric : type of metric for distance calculation
        self.centroids : the centroids
        self.centroids_labels: the centroids' labels
        '''
        self.k = n_clusters
        self.iterations = iterations
        self.metric = metric
        self.centroids = None
        self.centroids_labels = None

    def get_centroids(self,X,y):
      '''
      For every to be centroid, assign the values of a pre-existing sample from X
      The assigned sample values must be from a previously unused class/label, until all labels are used (repeat until K)
      Store and return the assigned centroids and their respective labels.
      '''
      

      centroids = np.array([])
      centroids_labels = np.array([])
      previous_labels = np.unique(y)
      for i in range(self.k):
        if len(previous_labels) == 0:
          previous_labels = np.unique(y)
        random_class = np.random.choice(previous_labels,1)
        previous_labels = np.delete(previous_labels,np.where(previous_labels == random_class)) #remove the class from the possible choices of class
        index = np.random.choice(np.where(y == random_class)[0],1) #get a random index where y == j
        sample_to_append = X[index]
        if len(centroids)==0:
          centroids = sample_to_append
        else:
          if np.asarray((sample_to_append == centroids).all(1)).any():
            possible_samples = X[y == random_class]
            appended = False
            for p_s in possible_samples:
              if not appended:
                if np.asarray((p_s == centroids).all(1)).any():
                  pass
                else:
                  sample_to_append = p_s
                  appended = True
            if not appended:
              sample_to_append = X[index]
          centroids = np.vstack((centroids,sample_to_append))
        centroids_labels = np.append(centroids_labels,random_class)
      return centroids,centroids_labels

    def euclidean_distance(self,a,b):
      '''
      Calculates the euclidean distance between the 'a' samples
      and the 'b' centroids' set of features

      Returns the euclidean distances (square root of sum of (a-b).T @ (a-b))
      '''
      distances = []
      a = np.atleast_2d(a)
      b = np.atleast_2d(b)      
      for sample in a:       
        distances.append(np.sqrt(np.sum(np.square(sample - b),axis=1)))
      return distances

    def get_distances(self,X,centroids):
      '''
      Calculates the distances of the X samples to the k centroids
      Returns the obtained distances
      '''
      distances = np.zeros((X.shape[0],self.k)) #Vetor 0 [Num samples, Num Centroids]
      for k in range(self.k): #For each centroid
        for i in range(len(distances[:,k])): #For each sample
          distances[i,k] = self.euclidean_distance(X[i],centroids[k,:])[0]
      return distances
    
    def get_labels(self,distances,X=None):
      '''
      Find the respective labels of the X samples based on their distance to the centroid

      Samples are labeled/clusterized based on the closest centroid to them.

      Returns the obtained labels
      '''
      labels = np.argmin(distances,axis=1).astype(str) #Get the closest centroids
      indexes = []      
      for i in range(len(self.centroids_labels)):
        indexes.append(np.where(labels == str(i)))
        labels = np.where(labels == str(i), self.centroids_labels[i],labels)
      if X is not None:
        return np.asarray(labels),indexes
      else:
        return np.asarray(labels)
    
    def get_clusters(self,distances,X):
      '''
      Find the respective cluster of the X samples based on their distance to the centroid

      Samples are labeled/clusterized based on the closest centroid to them.

      Returns the obtained clusters
      '''
      labels = np.argmin(distances,axis=1).astype(str) #Get the closest centroids
      indexes = []
      clusters = []      
      for i in range(len(self.centroids_labels)):
        indexes.append(np.where(labels == str(i)))
        labels = np.where(labels == str(i), self.centroids_labels[i],labels)
        clusters.append(X[indexes[i]])
        
      return clusters

    def compute_centroids(self, X, y,indexes):
      '''
      Creates a zero'd MATRIX k,j
      where k == number of centroids
      where j == number of features

      For every kth centroid, each jth feature's value  equals to
      the mean of all the samples within the kth cluster 
      '''
      centroids = np.zeros((self.k, X.shape[1]))
      for k in range(self.k):
        if len(X[indexes[k]]) == 0:          
          centroids[k] = self.centroids[k]
        else:
          mean = np.mean(X[indexes[k]],axis=0)
          centroids[k] = np.mean(X[indexes[k]],axis=0)
      return centroids

    def train(self,X,y):
        '''
        Trains the Algorithm to able to clusterize samples nearest to the K centroids
        Training is done with the following steps:
        
        1. From all the samples of the X training set, choose k samples to become centroids
        2. Store the centroids between the iterations
        3. Calculate the distances between the samples and the centroids
        4. Clusterize the samples nearest to each kth centroid, creating k clusters
        5. Update the centroids value with the mean of it's cluster's elements
        6. Stop if:
            -the centroids value diference between the iterations didn't change or is too small
             (rtol=1e-05, atol=1e-08, absolute(a - b) <= (atol + rtol * absolute(b)))
            -concluded the last iteration
        '''
        if X is None:
          raise ValueError("Expected at least one training sample. None was given.")
        if y is None:
          raise ValueError("Expected at least one training label. None was given.")

        self.centroids,self.centroids_labels = self.get_centroids(X,y)
        for iteration in range(self.iterations):
          last_centroids = self.centroids
          distances = self.get_distances(X, last_centroids)
          labels,indexes = self.get_labels(distances,X)
          self.centroids = self.compute_centroids(X, labels,indexes)
          if np.allclose(last_centroids,self.centroids):
            #print(f"Stopped training at the [{iteration+1}º] iteration")
            break
        #if iteration == (self.iterations-1):
        #  print(f"Stopped training at the [{iteration+1}º] iteration")
        
    def predict(self, X):
      '''
      Predicts the classes of a X dataset's samples based on their distances to
      the centroid(s)

      The obtained class is the same "class" as the nearest centroid

      Returns the obtained classes for the respective classified samples
      '''
      if self.metric == "euclidean":
        distances = self.get_distances(X, self.centroids)
        predictions = self.get_labels(distances)
        return predictions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
      else:
        raise ValueError("Expected valid metric type for calculating distances. Received: ",self.metric)
    def predict_clusters(self,X):
      '''
      Creates clusters from a dataset's samples based on their distances to
      the centroid(s)

      Returns the obtained clusters
      '''
      if self.metric == "euclidean":
        distances = self.get_distances(X, self.centroids)
        clusters = self.get_clusters(distances,X)
        return clusters
      else:
        raise ValueError("Expected valid metric type for calculating distances. Received: ",self.metric)
        
class NaiveBayes():
  def __init__(self):
    self.classes = None
    self.mean = None
    self.std = None
    self.prioris = None

  def train(self,X,y):
    #Get the number of (unique) classes
    self.classes = np.unique(y)
    n_classes = len(self.classes)

    #Get how many samples and features there are
    n_samples = X.shape[0]
    n_features = X.shape[1]  

    # mean, variance, priori para cada classe
    self.mean = np.zeros((n_classes, n_features))
    self.std = np.zeros((n_classes, n_features))
    self.prioris = np.zeros(n_classes)

    #For every class, calculate mean, variance and priori
    for index, cls in enumerate(self.classes):
      x_classe = X[y == cls]
      self.mean[index, :] = x_classe.mean(axis=0)
      self.std[index, :] = x_classe.std(axis=0)
      self.prioris[index] = x_classe.shape[0]/(n_samples)
  
  #Calculates de Posterioris Probabilities
  def posteriori(self, x):
    posterioris = []
    
    #For every class, compute likelihood/pdf, get priori and compute posteriori
    for index, cls in enumerate(self.classes):
      likelihood = self.pdf(index,x)
      probability = self.prioris[index]
      for xy in likelihood:
        probability *= xy
      posterioris.append(probability)     
    return posterioris


  def predict(self,X):
    predictions = []
    iterator = 0
    #For every sample
    for x in X:
      #Find Posterioris
      probability = self.posteriori(x)
      
      #Label is the class with the highest probability
      predictions.append(self.classes[np.argmax(probability)])
      iterator+=1
    return np.asarray(predictions)
  
  #Probability Density Function, PDF.
  def pdf(self,class_index,x):
    mean = self.mean[class_index]
    deviation = self.std[class_index]
    deviation [deviation == 0] = 1e-10
    self.std[class_index] = deviation
    likelihood = (1/(deviation * np.sqrt(2 * np.pi))) * np.exp(-((x - mean)**2)/(2*(deviation**2)))
    return likelihood

class LinearBayesClassifier():
  def __init__(self,tipo=''):
    self.classes = None
    self.mean = None
    self.covariance_matrix = None
    self.noise_matrix = None
    self.prioris = None
    self.tipo = tipo

  def train(self,X,y):
    #Get the number of (unique) classes
    self.classes = np.unique(y)
    n_classes = len(self.classes)
    
    #Get how many samples and features there are
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    #Create Noise Matrix
    self.noise_matrix = np.eye(X.shape[1]) * 1e-8
    
    # mean, variance, priori para cada classe
    self.mean = np.zeros((n_classes, n_features))
    self.prioris = np.zeros(n_classes)

    #For every class, calculate mean, variance and priori
    for index, cls in enumerate(self.classes):
      x_classe = X[y == cls]
      self.mean[index, :] = x_classe.mean(axis=0)
      self.prioris[index] = x_classe.shape[0]/(n_samples)
    self.covariance_matrix = self.covar_matrix(X, n_samples, n_features)
  
  #Calculates the single covariance matrix
  def covar_matrix(self, X, n_samples, n_features):
    cov_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        mean_i = np.mean(X[:, i])
        for j in range(n_features): 
            mean_j = np.mean(X[:, j])
            cov_matrix[i, j] = np.sum((X[:, i] - mean_i) * (X[:, j] - mean_j)) / (n_samples - 1)
    #If a non-default option for the covariance matrix was chosen
    if self.tipo == 'diagonal' or self.tipo == 'reversed_diagonal':
      return self.get_special_matrix(cov_matrix)
    else:
      return cov_matrix
  
  #Calculates special matrix
  def get_special_matrix(self,cov_matrix):
    new_matrix = np.zeros((cov_matrix.shape))
    #Diagonal Matrix, with zeroes below and above the main diagonal.
    if self.tipo == 'diagonal':
      for i in range(cov_matrix.shape[0]):
        new_matrix[i][i] = cov_matrix[i][i]
    #Secondary Diagonal Matrix with Main Diagonal set to 1 (if original value positive) or -1 (if negative)
    elif self.tipo == 'reversed_diagonal':
      for i in reversed(range(cov_matrix.shape[0])):
        new_matrix[cov_matrix.shape[0] - i - 1][i] = cov_matrix[i][i]
        if cov_matrix[i][i] > 0:
          new_matrix[i][i] = 1
        elif cov_matrix[i][i] < 0:
          new_matrix[i][i] = -1
    return new_matrix

  #Calculates de Posterioris Probability
  def posteriori(self,X):
    posterioris = []
    
    #For every class, compute likelihood/pdf, get priori and compute posteriori
    for index, cls in enumerate(self.classes):
      likelihood = self.multivar_gaussian(X,self.mean[index],self.covariance_matrix)
      probability = likelihood * self.prioris[index]
      posterioris.append(probability)     
    return posterioris

  #Predict
  def predict(self,X):
    predictions = []
    iterator = 0
    #For every sample
    for x in X:
      #Find Posterioris
      probability = self.posteriori(x)
      
      #Label is the class with the highest probability
      predictions.append(self.classes[np.argmax(probability)])
      iterator+=1
    return np.asarray(predictions)
  
  def multivar_gaussian(self, x, mean, covar):
    determinant = np.linalg.det(covar)
    if determinant == 0:
      determinant = 1
      covar = covar + self.noise_matrix
    numerator = np.exp(-(1/2) * np.dot(np.dot((x - mean), np.linalg.inv(covar)), (x - mean)))
    denominator = 1 / (((2*np.pi)**(len(x)/2))*(determinant**(1/2)))
    return numerator * denominator

class QuadraticBayesClassifier():
  def __init__(self,tipo=''):
    self.classes = None
    self.mean = None
    self.covariance_matrix = None
    self.noise_matrix = None
    self.prioris = None
    self.tipo = tipo

  def train(self,X,y):
    #Get the number of (unique) classes
    self.classes = np.unique(y)
    n_classes = len(self.classes)
    
    #Get how many samples and features there are
    n_samples = X.shape[0]
    n_features = X.shape[1]

    #Create Noise Matrix
    self.noise_matrix = np.eye(X.shape[1]) * 1e-8
    
    # mean, variance, priori para cada classe
    self.mean = np.zeros((n_classes, n_features))
    self.prioris = np.zeros(n_classes)

    #For every class, calculate mean, variance and priori
    for index, cls in enumerate(self.classes):
      x_classe = X[y == cls]
      self.mean[index, :] = x_classe.mean(axis=0)
      self.prioris[index] = x_classe.shape[0]/(n_samples)
    self.covariance_matrix = self.calc_cov(X,y)

  #Calculates the original matrix of the classes
  def calc_cov(self,X,y):
    cov_matrix = []
    for index, classe in enumerate(self.classes):
      x_classe = X[y == classe]
      cov_matrix.append(self.covar_matrix(x_classe,x_classe.shape[0],x_classe.shape[1]))
    #If another type of matrix other than the default option was chosen
    if self.tipo == 'diagonal':
      return  self.get_diagonal_matrix(cov_matrix)
    elif self.tipo == 'average':
      return self.get_average_matrix(cov_matrix)
    return cov_matrix
  
  #Auxiliary function for calculating covariance matrixes
  def covar_matrix(self, X, n_samples, n_features):
    cov_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        mean_i = np.mean(X[:, i])
        for j in range(n_features): 
            mean_j = np.mean(X[:, j])
            cov_matrix[i, j] = np.sum((X[:, i] - mean_i) * (X[:, j] - mean_j)) / (n_samples - 1)
    return cov_matrix
  
  #Get Special 'Diagonal' Matrix, with only the main diagonal's elements
  #with values while the others are set to zeroes
  def get_diagonal_matrix(self,cov_matrixes):
    for y,cov_matrix in enumerate(cov_matrixes):
      new_matrix = np.zeros((cov_matrix.shape))
      for i in range(cov_matrix.shape[0]):
        new_matrix[i][i] = cov_matrix[i][i]
      cov_matrixes[y] = new_matrix 
    return cov_matrixes

  #Calculates the Average Matrix from the Classes Matrixes, sums the average matrix with the class matrix,
  #divides it by two and then returns it
  def get_average_matrix(self,cov_matrixes):
    average_cov_matrix = cov_matrixes[0]
    for cov_mat in cov_matrixes[1:]:
      average_cov_matrix += cov_mat
    average_cov_matrix = average_cov_matrix/len(cov_matrixes)  
    for i in range(len(cov_matrixes)):
      cov_matrixes[i] = (cov_matrixes[i] + average_cov_matrix)/2
    return cov_matrixes
  
  #Calculates the Posterioris Probability
  def posteriori(self,X):
    posterioris = []
    #For every class, compute likelihood/pdf, get priori and compute posteriori
    for index, cls in enumerate(self.classes):
      likelihood = self.multivar_gaussian(X,self.mean[index],self.covariance_matrix[index])
      probability = likelihood * self.prioris[index]
      posterioris.append(probability)     
    return posterioris
  
  #Predict
  def predict(self,X):
    predictions = []
    #For every sample
    iterator = 0
    for x in X:
      probability = self.posteriori(x)
      predictions.append(self.classes[np.argmax(probability)])
      iterator+=1
    return np.asarray(predictions)
  
  #Calculates the gaussian pdf.
  def multivar_gaussian(self, x, mean, covar):
    determinant = np.linalg.det(covar)
    if determinant == 0:
      determinant = 1
      covar = covar + self.noise_matrix
    numerator = np.exp(-(1/2) * np.dot(np.dot((x - mean), np.linalg.inv(covar)), (x - mean)))
    denominator = 1 / (((2*np.pi)**(len(x)/2))*(determinant**(1/2)))
    return numerator * denominator