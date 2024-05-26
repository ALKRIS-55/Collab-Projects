# Collab-Projects

## Self-Driving Car Using Udacity's Car Simulator and Deep Neural Networks
**Overview**
- **Libraries:** Utilizes TensorFlow, Keras, OpenCV, Pandas, NumPy, and imgaug for data augmentation and model training.
- **Data Handling:** Processes driving log data, applies image augmentation (zoom, pan, brightness adjustment, flipping).
- **Model Architecture:** Based on NVIDIA’s self-driving car model with convolutional layers and dropout for regularization.
- **Training:** Uses Adam optimizer, mean squared error loss, and fit generator for training and validation data.
- **Output:** Trained model predicts steering angles, saved as model.h5.
- **Project link:** [Self-Driving Car Project](https://colab.research.google.com/drive/1nxco8RgT_wpQj4ezQsa9BLcfMnX--1hg?usp=sharing)

## Sentiment Analysis for Dow Jones (DJIA) Stock Using Newspaper Headlines Dataset
**Overview**
- **Dataset:** Loaded and explored dataset with 4101 rows and 27 columns (Date, Label, Top1-Top25).
- **Visualization:** Plotted the count of 'Label' column (0 - Stock down/same, 1 - Stock up).
- **Data Cleaning:** Dropped NaN values, split into train and test sets.
- **Text Preprocessing:** Removed punctuation, converted to lowercase, and applied stemming to headlines.
- **Modeling:** Created embeddings of new headline to capture semantic meaning.
- **Algorithms:**
  - Logistic Regression: Accuracy 85.98%, Precision 0.87, Recall 0.85.
  - Random Forest: Accuracy 84.39%, Precision 0.84, Recall 0.86.
  - Naive Bayes: Accuracy 83.86%, Precision 0.85, Recall 0.83.
- **Evaluation:** Confusion matrices and heatmaps for each model.
- **Project link:** [Sentiment Analysis Project](https://colab.research.google.com/drive/1ipkMaSJcg_lpyxG6JuL3ggV9KW-nfgZI?usp=sharing)

## Dropout Prediction
**Overview**
-**Loading and Understanding the Data:**
- The dataset is loaded using pandas, and its basic structure is examined.
- Information about the columns, their data types, and non-null counts is obtained.
- The count of missing values in each column is calculated and found to be zero.
-**Data Preprocessing:**
- The target variable, initially represented as 'Graduate', 'Dropout', and 'Enrolled', is transformed into numerical labels (0, 1, 2) for classification purposes.
- Histograms are generated to visualize the distributions of numeric features.
-**Feature Selection and Correlation Analysis:**
- Correlation coefficients between features and the target variable are calculated.
- The most correlated features with the target variable are identified for further analysis.
-**Scaling the Dataset:**
- The dataset is standardized using the StandardScaler to bring all features to the same scale, which is a common preprocessing step in many machine learning algorithms.
-**Splitting the Dataset:**
- The dataset is split into training, validation, and test sets using train_test_split.
- The stratify parameter is used to maintain the distribution of the target variable in each split.
-**Model Building and Evaluation:**
- Various machine learning models are trained and evaluated, including K-Nearest Neighbors (KNN).
- Hyperparameter tuning is performed to find the optimal number of neighbors for the KNN algorithm.
- Cross-validation techniques are employed to assess the models' performance and generalization ability.
- **Project link:** [Dropout Prediction Project](https://colab.research.google.com/drive/1w2uQdXLjNh4inV-eLH1PmpjHs_ZGwmCH?usp=sharing)
