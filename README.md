# Classification-of-Cats-and-Dogs-using-Support-Vector-Machine-SVM-
Support Vector Machines (SVM) are a set of supervised learning methods used for classification, regression, and outliers detection. They work by finding the optimal hyperplane that best separates different classes in a dataset. Support Vector Classifier (SVC) is a specific implementation of SVM focused on classification tasks, aiming to maximize the margin between the data points of different classes.

# Preprocessing the dataset

import os

import cv2

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

#Constants

IMG_SIZE = 64

def preprocess_images(folder):

    images = []
    labels = []

    for filename in os.listdir(folder):
    
        if filename.endswith('.jpg'):
        
            label = 1 if 'dog' in filename else 0
            
            img = cv2.imread(os.path.join(folder, filename))
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            img = img / 255.0  # Normalization
            
            images.append(img.flatten())  # Flattening
            
            labels.append(label)

    return np.array(images), np.array(labels)
    
train = '/content/drive/MyDrive/Colab Notebooks/dogs-vs-cats/train/train1'

#Preprocess training data

X_train, y_train = preprocess_images(train)

#Split train data into train and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Training Data Shape:", X_train.shape)

print("Validation Data Shape:", X_val.shape)


# Training the model using non-linear kernel RBF  

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss

#Training the SVM model

svm_model = SVC(probability=True, kernel='rbf')

svm_model.fit(X_train, y_train)

#Predicting on validation set

y_pred = svm_model.predict(X_val)

y_prob = svm_model.predict_proba(X_val)[:, 1]

# Evaluation metrics (Classification_Report, Confusion_matrix, Roc_Auc_score, Log_loss)

print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

print("Classification Report:\n", classification_report(y_val, y_pred))

print("ROC AUC Score:", roc_auc_score(y_val, y_prob))

print("Log Loss:", log_loss(y_val, y_prob))

# Results:

Confusion Matrix:

 [[335 170]
 [176 320]]
 
Classification Report:

               precision    recall  f1-score   support

           0       0.66      0.66      0.66       505
           
           1       0.65      0.65      0.65       496

    accuracy                           0.65      1001
    
   macro avg       0.65      0.65      0.65      1001
   
weighted avg       0.65      0.65      0.65      1001

ROC AUC Score: 0.7295572500798466

Log Loss: 0.6071978955734438

