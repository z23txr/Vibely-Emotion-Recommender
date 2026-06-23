# Vibely - Emotion-Based Movie Recommender 

![Vibely Banner](background.webp) 

Vibely is an interactive AI-powered web application that recommends movies based on your real-time facial expressions. Forget scrolling endlessly through lists let your mood decide your next movie!

##  Features
* **Real-time Emotion Detection**: Captures a quick photo via your webcam and uses Deep Learning to detect your current emotion (Happy, Sad, Angry, Surprised, etc.).
* **AI-Powered Recommendations**: Uses TF-IDF and Cosine Similarity to find the best matching movies based on genres, overviews, and your specific mood.
* **Manual Search & Filters**: If you don't want to use the camera, you can search for movies manually or browse by genre.
* **Beautiful UI/UX**: Built with a sleek, modern, glassmorphism-inspired dark mode interface.
* **Cloud Ready**: Deployed on Streamlit Community Cloud for instant access anywhere.

##  Tech Stack
* **Frontend/UI**: Streamlit, Custom CSS, HTML
* **Computer Vision**: OpenCV (Haar Cascades for face detection)
* **Deep Learning**: TensorFlow / Keras (CNN for emotion classification)
* **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)
* **Data Manipulation**: Pandas, NumPy

##  Live Demo
You can try out the live application here:(https://vibely-emotion-recommender.streamlit.app/)

##  Running Locally

To run this project on your local machine, follow these steps:

### Prerequisites
Make sure you have Python 3.10 or 3.11 installed.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Vibely-Emotion-Recommender.git
cd Vibely-Emotion-Recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app


## 📂 Project Structure
* `app.py`: The main Streamlit web application.
* `app/core/emotion_detector.py`: Contains the logic for the Deep Learning model.
* `emotion_model.h5`: Pre-trained CNN model for facial emotion recognition.
* `fully_cleaned_movies.csv`: Dataset containing movie titles, genres, and overviews.
* `haarcascade_frontalface_default.xml`: OpenCV model for face detection.
* `requirements.txt`: List of dependencies needed to run the app.

