# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

# http://scikit-learn.sourceforge.net/dev/tutorial/basic/tutorial.html#introduction
"""scikit-learn comes with a few standard datasets, for instance the iris and digits datasets for classification and the boston house prices dataset for
regression.
"""
from sklearn import datasets
digits = datasets.load_digits()

# <codecell>

"""In general, a learning problem considers a set of n samples of data and trys to predict properties of unknown data. If each sample is more than a single number, is it said to have several attributes, or features.

A dataset is a dictionary-like object that holds all the data and some metadata about the data. This data is stored in the .data member, which
is a 2d array of n_samples, n_features. In the case of supervised problem, explanatory variables are stored in the .target member.
"""

# <codecell>

"""digits.data gives access to the features that can be used to classify the digits samples
"""
print digits.data

# <codecell>

"""digits.target gives the classification for the digit dataset, that is the number corresponding to each digit image that we are trying to learn
"""
digits.target

# <codecell>

"""The data is always a 2D array, n_samples, n_features, although the original data may have had a different shape. In the case of the digits, each
original sample is an image of shape 8, 8 and can be accessed using
"""
import pylab as pl
print digits.images[0]
pl.axis('off')
pl.imshow(digits.images[0], cmap=pl.cm.gray_r, interpolation='nearest')
pl.show()
print digits.target[0]

# <codecell>

"""In the case of the digits dataset, the task is to predict the value of a hand-written digit from an image. We are given samples of each of the 10
possible classes on which we fit an estimator to be able to predict the labels corresponding to new data.
An example of estimator is the class sklearn.svm.SVC that implements Support Vector Classification. The constructor of an estimator takes as arguments
the parameters of the model, but for the time being, we will consider the estimator as a black box
"""
from sklearn import svm

# <codecell>

"""In this example we set the value of gamma manually. It is possible to automatically find good values for the parameters by using tools such as grid
search and cross validation.
"""
clf = svm.SVC(gamma=0.001, C=100.)

# <codecell>

"""It now must be fitted to the model, that is, it must learn from the model. This is done by passing our training set to the fit method. As a training
set, let us use all the images of our dataset apart from the last one

We create a predictor...
"""
clf.fit(digits.data[:-10], digits.target[:-10])

# <codecell>

"""Now you can predict new values, in particular, we can ask to the classifier what is the digit of our last image in the digits dataset, which we have
not used to train the classifier"""
for i in range(10):
    print clf.predict(digits.data[-i]), digits.target[-i]
    pl.axis('off')
    pl.imshow(digits.images[-i], cmap=pl.cm.gray_r, interpolation='nearest')
    pl.show()

# <codecell>

"""However this is not a really good way to split our dataset in practice. Let's try out some cross-validation which will give us more confidence in our model."""
from sklearn.cross_validation import cross_val_score
clf = svm.SVC(gamma=0.001, C=100.)
scores = cross_val_score(clf, digits.data, digits.target, cv=5)

print scores
print scores.mean()

# <codecell>


