# Parkinsons-prediction--ML-Voice-features
In this project, a machine learning-based predictive system was developed to classify individuals as Parkinsonian or healthy based on voice recordings. 
The study utilized a public dataset sourced from Kaggle, which contains biomedical voice measurements from patients diagnosed with Parkinson’s disease as well as from healthy individuals. 
Key acoustic features such as jitter, shimmer, fundamental frequency (F0), Pitch Period Entropy (PPE), and noise-to-harmonics ratios (NHR) were used to train ML models including Random Forest and K-Nearest Neighbors (KNN). 
These features are known to capture the instability and irregularities in vocal cord vibrations that are often associated with Parkinson’s.

Tools:
Python 3.8 (Jupyter Notebook and Visual Studio Code)
Libraries:
1. Data Handling & Manipulation
pandas – Used to handle and organize tabular data for analysis.
numpy – Used for performing fast numerical operations on arrays.
2. Data Visualization
matplotlib – Used to create basic plots and charts for data understanding.
seaborn – Used to make attractive statistical graphs like heatmaps.
plotly – Used to create interactive and dynamic visualizations.
3. Data Preprocessing
sklearn.preprocessing – Used to scale, normalize, and encode data before modeling.
sklearn.model_selection – Used to split data and perform cross-validation.
sklearn.impute – Used to fill missing values in the dataset.
4. Machine Learning Models
sklearn.ensemble – Used to apply models like Random Forest for classification.
sklearn.linear_model – Used to build simple and interpretable models like logistic regression.
sklearn.svm – Used to apply Support Vector Machines for accurate classification.
sklearn.tree – Used to create decision trees for easy model interpretation.
sklearn.neighbors – Used for classification using K-Nearest Neighbors algorithm.
5. Model Evaluation & Performance Metrics
sklearn.metrics – Used to evaluate model performance with metrics like accuracy and F1-score.
6. Saving & Loading Models
joblib – Used to save and load trained machine learning models.
7. Audio Processing & Feature Extraction
librosa – Used to extract audio features like MFCCs and pitch.
parselmouth – Used to extract voice features like jitter, shimmer, and NHR.
soundfile – Used to read and write audio files.
8. Web App Deployment & Interface
gradio – Used to create an interactive web app for real-time predictions.

**RESULTS**
Model Accuracy Score K-Nearest Neighbours (KNN) 94.8%
Decision Tree Algorithm (DTA)- 92.3%
Support Vector Machine (SVM) 87.1%
Random Forest Classifier (RFC)- 94.9% 
Logistic Regression Model (LRM) 89.7%

**RUN**
model_script.py for model training
app.py for user interface

parkinsons_model.pkl- model made
extreacted_features.py- extracts the features from user input audio file
