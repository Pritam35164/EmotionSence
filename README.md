## Human Emotion Detection from Voice
# Project Overview

Human emotions are deeply expressed through voice — whether we're happy, sad, calm, or angry, our tone reveals it.
This project uses Machine Learning and Audio Signal Processing to automatically detect a speaker’s emotional state using their voice.

✅ Upload or record an audio file

✅ Instantly get the detected emotion & confidence score

✅ View emotional trends over time in a live graph

Built using Python, Librosa, Scikit-learn, and Streamlit, this application demonstrates the power of AI in understanding human emotion for real-world applications such as mental health monitoring, customer support analytics, and user-personalized systems.

#Features
Upload or record voice in real-time

Extracts advanced audio features (MFCC, Chroma, Spectral Contrast)

Predicts emotion using a trained SVM classifier

Shows confidence score of prediction

Displays a dynamic emotion trend graph based on user session history

Saves prediction logs for future analysis

# Tech Stack

Component	Technology Used

Programming	Python

ML Model	SVM (Support Vector Machine)

Audio Processing	Librosa

GUI / Web App	Streamlit

Dataset	RAVDESS Emotional Speech Dataset

# Project Structure

emotion-detection-voice/
│
├── features_mfcc_delta.csv          # Extracted features
├── generalized_emotion_model.joblib # Trained SVM model
├── label_encoder.pkl                # Encoded emotion labels
├── app.py                           # Streamlit UI file
├── train_generalized_model.ipynb    # Model training notebook
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies

# Emotion Classes

The model is trained to recognize the following emotions:

😌 Calm       🙂 Happy       😢 Sad       😡 Angry
😨 Fearful    🤢 Disgust     😲 Surprised 😐 Neutral

# Installation & Setup

🔰 Step 1: Clone the Repository
git clone https://github.com/your-username/emotion-detection-voice.git

cd emotion-detection-voice

🔧 Step 2: Install Dependencies

pip install -r requirements.txt

▶️ Step 3: Run the App

streamlit run app.py

🎚️ How It Works

User uploads or records a voice sample

Features such as MFCC and Chroma are extracted

The pre-trained SVM model predicts the emotion

The confidence score is displayed

Prediction is added to the emotion trend graph

# Sample Model Performance

Emotion	Precision	Recall	F1-Score

Angry	0.90	0.98	0.94

Happy	0.89	0.88	0.89

Calm	0.86	0.89	0.88

Overall Accuracy	0.90	-	-

Cross-Validation Accuracy: 97%

# Future Enhancements
Add real-time microphone streaming

Switch to deep learning models (CNN/LSTM) for better accuracy

Deploy as mobile/web API

Enable multi-language emotion detection


