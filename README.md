SMS Spam Detection

    Objective:
        -The objective is to develop an SMS classification model that identifies spam messages.

    Dataset Details:
        -The dataset consists of labeled messages categorized as spam or non-spam.
        -It is preprocessed to remove noise and convert text into numerical representations.

    Working Process:

        Data Preprocessing:
            -Remove special characters and stopwords.
            -Convert text to lowercase.
            -Apply vectorization (TF-IDF, CountVectorizer).

        Model Training:
            -Split data into training and testing sets.
            -Train different machine learning models.

        Model Evaluation:
            -Compare accuracy, precision, recall, and F1-score.
            -Select the best-performing model.

        Prediction and Deployment:
            -Test the model with new SMS messages.
            -Deploy the model using Flask or another framework.

Models Used

        >Naïve Bayes
        >Support Vector Machine (SVM)
        >Random Forest Classifier
        >Logistic Regression
        >Gradient Boosting Classifier
        >K-Nearest Neighbors (KNN)
        >Multinomial Naive Bayes
        >RNN
        >LSTM


Outputs

    -Accuracy and performance metrics for each model.
    -Classification report with precision, recall, and F1-score.
    -A deployed model that can classify new SMS messages as spam or non-spam.

Software Requirements
    -Python 3
    -Jupyter Notebook