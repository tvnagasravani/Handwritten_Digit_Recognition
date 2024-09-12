# Handwritten Digit Recognition

## Introduction
Every human being has unique handwriting. While some handwriting is easy to understand, others may not be as legible. In fact, there is a wide variation in how individuals write letters and digits. To address this variation, we require a system capable of recognizing handwritten digits, regardless of how differently people write them.

This project focuses on handwritten digit recognition, which involves detecting and identifying digits written by various individuals. The goal is to create a machine learning application that can accurately interpret handwritten digits.

## Applications and Advantages of Handwritten Digit Recognition
Handwritten digit recognition has a wide range of applications across various industries, such as:

- **Finance**
- **Retail Industry** – for fast processing of documents
- **Insurance and Banking Sectors**
- **Healthcare**
- **Logistics Companies**

It is a crucial tool that converts handwritten digits into machine-readable data, streamlining processes in several sectors.

## Dataset
To detect handwritten digits, a large and diverse dataset is required. The MNIST dataset is commonly used for this purpose as it provides a collection of handwritten digits in various formats.

- **MNIST** refers to the Modified National Institute of Standards and Technology database.
- This dataset consists of 60,000 training examples and 10,000 testing examples.
- The dataset contains 4 files:
  - Training set images
  - Training set labels
  - Test set images
  - Test set labels

## Implementation

### Steps to implement handwritten digit recognition:

1. **Download the Dataset**:
    - First, download the MNIST dataset file, which is in ZIP format.

2. **Extract the Dataset**:
    - After downloading, extract the dataset, which contains the files necessary for training and testing.

3. **Organize the Files**:
    - Inside the main project folder, you will find six different folders corresponding to six machine learning algorithms (both supervised and unsupervised). These include algorithms such as:
      - SVM
      - Logistic Regression
      - Linear Regression
      - K-Nearest Neighbors (KNN)
      - Random Forest, etc.

4. **Prepare the Dataset for Each Algorithm**:
    - Copy and paste the MNIST dataset into each of the six algorithm folders, as the dataset will be used for training and testing all models.

5. **Run the K-Nearest Neighbors (KNN) Algorithm**:
    - Open the `knn` folder and run the algorithm by opening the command prompt within this folder.
    - In the command prompt, run the following command:
      ```bash
      python knn.py
      ```

6. **Model Execution**:
    - The script will execute and display the confusion matrix for the training data, validation data, and testing data. It will also display the accuracy for each dataset.

7. **Repeat for Other Algorithms**:
    - Repeat steps 5 and 6 for all six algorithms (SVM, Logistic Regression, Random Forest, etc.).

### Note:
If you do not want the data to display in the command prompt, you can comment out specific lines (10, 11, 103, 104) in the `.py` files.

8. **Evaluate Accuracy**:
    - Compare the accuracy of each algorithm to determine the best one for handwritten digit recognition.

## Results

| Algorithm               | Accuracy     |
|-------------------------|--------------|
| K-Nearest Neighbors (KNN)| 97.88%       |
| SVM                     | 99.91%       |
| Random Forest            | 99.71%       |
| Logistic Regression      | 83.15%       |
| Linear Regression        | 30.88%       |

## Analysis

By analyzing the results, we can conclude the following:

- **Algorithm with the highest accuracy**: SVM (99.91%)
- **Algorithm with the lowest accuracy**: Linear Regression (30.88%)  
  - Note: The accuracy of Linear Regression can be improved using Logistic Regression.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
