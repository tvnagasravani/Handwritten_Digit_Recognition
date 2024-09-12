import sys
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

print('\nLoading MNIST Data...')

data = MNIST('./MNIST_Dataset_Loader/dataset/')

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

# Features
X = train_img

# Labels
y = train_labels

print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

print('\nLinear Regression Model')
print('\nPickling the Model for Future Use...')
model = LinearRegression()
model.fit(X_train, y_train)

with open('MNIST_LinearRegression.pickle', 'wb') as f:
    pickle.dump(model, f)

pickle_in = open('MNIST_LinearRegression.pickle', 'rb')
model = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Model...')
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)
confidence = accuracy_score(y_test, y_pred_rounded)

print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test, y_pred_rounded)

print('\nLinear Regression Model Confidence (Accuracy): ', confidence)
print('\nPredicted Values: ', y_pred_rounded)
print('\nConfusion Matrix: \n', conf_mat)

plt.matshow(conf_mat)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print('\nMaking Predictions on Test Input Images...')
test_labels_pred = model.predict(test_img)
test_labels_pred_rounded = np.round(test_labels_pred).astype(int)

print('\nCalculating Accuracy of Trained Model on Test Data...')
acc = accuracy_score(test_labels, test_labels_pred_rounded)

print('\nCreating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels, test_labels_pred_rounded)

print('\nPredicted Labels for Test Images: ', test_labels_pred_rounded)
print('\nAccuracy of Model on Test Images: ', acc)
print('\nConfusion Matrix for Test Data: \n', conf_mat_test)

plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

a = np.random.randint(1, 50, 20)
for i in a:
    two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
    plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[i], test_labels_pred_rounded[i]))
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    plt.show()
