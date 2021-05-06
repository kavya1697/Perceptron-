import numpy as np
import pandas as pd
import sys
import gzip
import shutil
from matplotlib import pyplot as plt


class Perceptron(object):  
    def __init__(self, eta=0.1, n_iter=10, random_state=1):                     #eta : float,Learning rate (between 0.0 and 1.0), #n_iter : interations, Passes over the training dataset.
                                                                                #random_state : int, Random number generator seed for random weight initialization

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# ### Reading-in the Iris data


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)

print(df)

# select setosa and versicolor
y = df.iloc[0:100, 1].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()



# ### Training the perceptron model

# In[12]:


ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)


plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.ylim([-0.1, max(ppn.errors_) + 0.1])		#optional: for the sake of a better visualisation
plt.xlabel('Epochs')
plt.ylabel('Number of errors')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' % kind)
    images_path = os.path.join(path, 
                               'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels




if (sys.version_info > (3, 0)):
    writemode = 'wb'
else:
    writemode = 'w'

zipped_mnist = [f for f in os.listdir('./') if f.endswith('ubyte.gz')]
for z in zipped_mnist:
    with gzip.GzipFile(z, mode='rb') as decompressed, open(z[:-3], writemode) as outfile:
        outfile.write(decompressed.read()) 




X_train, y_train = load_mnist('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

print('X_train data: ', X_train[0])
print('X_train label: ', y_train[0])



X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))







# ## Implementing a multi-layer perceptron
class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.

    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatche_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.

    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.

    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):

        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation

        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.

        Returns
        -----------
        onehot : array, shape = (n_samples, n_labels)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):                                                      #Compute logistic function (sigmoid)
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Compute forward propagation step"""

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h
        print('Dimensions of X: ', X.shape)
        print('Dimensions of self.w_h: ', self.w_h.shape)
        print('Dimensions of self.b_h: ', self.b_h.shape)
        print('Dimensions of z_h: ', z_h.shape)
        

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)
        print('Dimensions of a_h: ', a_h.shape)                                   # step 3: net input of output layer
        z_out = np.dot(a_h, self.w_out) + self.b_out
        print('Dimensions of a_h: ', a_h.shape)                                    # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        print('Dimensions of self.w_out: ', self.w_out.shape)
        print('Dimensions of self.b_out: ', self.b_out.shape)
        print('Dimensions of z_out: ', z_out.shape)                              
                                                                                 # -> [n_samples, n_classlabels]
        
        

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)
        print('Dimensions of a_out: ', a_out.shape)

        return z_h, a_h, z_out, a_out



    def fit(self, X_train, y_train, X_valid, y_valid):                            #Learn weights from training data.
        n_output = np.unique(y_train).shape[0]                                     # number of class labels
        n_features = X_train.shape[1]

  # Weight initialization#
        
        self.b_h = np.zeros(self.n_hidde)                                         # weights for input -> hidden
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        
        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        
        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])


            if self.shuffle:
                self.random.shuffle(indices)


            print('Minibatch size: ', self.minibatch_size)
            print('Total number of training data points: ', indices.shape[0])
            print('end index of the following range function: ', indices.shape[0] - self.minibatch_size)
            for start_idx in range(0, indices.shape[0] - self.minibatch_size, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                
                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])


                break #It is practically useless to establish a for loop with a break statement which is not conditioned. We use it here only since we are yet to continue working with the same class next lab. 
                


            break   #If you haven't already, please read the comment on the break statement above. It also applies here. 
            
            # Evaluation after each epoch during training  #remove
            
	
        return self                                      
            
                                                                                 
n_epochs = 4 

nn = NeuralNetMLP(n_hidden=50, 
                  l2=0.01, 
                  epochs=n_epochs, 
                  eta=0.1,
                  minibatch_size=100, 
                  seed=1)

nn.fit(X_train=X_train[:55000], 
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])




