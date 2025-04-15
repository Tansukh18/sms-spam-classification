# sms-spam-classification 

What I accomplished : Developed and deployed a real-time SMS spam classifier web app that predicts whether a given message is "Spam" or "Ham" with over 97% accuracy.

What I did : Preprocessed a dataset of labeled SMS messages using techniques such as lowercasing, punctuation removal, tokenization, stopword removal, and TF-IDF vectorization. Trained and evaluated multiple machine learning models including Naïve Bayes, Logistic Regression, and SVM using scikit-learn. Selected the best-performing model and integrated it into an interactive web interface using Streamlit. The app allows users to input custom SMS text and receive instant classification results.

Impact/Result : Delivered a lightweight, responsive web-based solution for identifying spam messages, which can be adapted for email or chat filtering systems and used for educational or practical anti-spam purposes.

Real-World Problem: With the massive increase in unsolicited and fraudulent messages, spam detection is essential to protect users from scams, phishing, and data breaches. Manual filtering is inefficient and often ineffective.

Solution Provided: This project provides an automated, intelligent SMS classification system accessible through a simple web interface. It helps users quickly identify spam messages, reducing the risk of engaging with malicious content.

Technical Tools Used:
Programming Language: Python
Machine Learning: Scikit-learn (Naïve Bayes, Logistic Regression, SVM)
Text Processing: NLTK, Regular Expressions, TF-IDF
Web Deployment: Streamlit
Data Visualization: Matplotlib, Seaborn
Model Evaluation: Accuracy, Confusion Matrix, Precision, Recall, F1-score

Dataset Used: UCI SMS Spam Collection Dataset
