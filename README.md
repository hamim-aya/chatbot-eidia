EIDIA ChatBot: Neural Intent Recognition
This project implements a sophisticated Intent Recognition System designed for conversational AI. It utilizes a neural network architecture to classify user queries into predefined intents, enabling a chatbot to respond accurately to a wide range of academic and administrative inquiries.

🚀 Features
Neural Classification: Uses a deep learning model to map user input to specific intents with high accuracy.

Natural Language Processing: Implements tokenization and text preprocessing to handle diverse human language patterns.

Scalable Knowledge Base: Organized via a structured intent system (JSON/Patterns) for easy updates to the chatbot's "intelligence".

Hybrid Architecture: Capable of handling both direct pattern matching and probabilistic neural responses.

🛠️ Project Structure
Based on your repository, the core components are:

app.py: The main entry point (likely a Flask or FastAPI web interface).

chatbot_model.h5: The trained neural network weights.

tokenizer.pkl & label_encoder.pkl: Preprocessing artifacts for text vectorization.

hybrid_chatbot.py: Logic combining neural inference with fallback mechanisms.

📋 Prerequisites
Ensure you have Python 3.8+ installed. You can install the necessary dependencies using:

Bash
pip install -r requirements.txt
Note: Major dependencies include TensorFlow/Keras for the model and NLTK/Spacy for text processing.

🚦 How to Run
Clone the repository:

Bash
git clone https://github.com/hamim-aya/chatbot-eidia.git
cd chatbot-eidia
Start the Application:

Bash
python app.py
Interact: Open your browser to the local address provided (usually http://127.0.0.1:5000) to start chatting with the AI.

🧠 Model Training
The model was trained on a dataset of intent patterns. The training process involved:

Preprocessing: Lowercasing, removing punctuation, and tokenizing words.

Vectorization: Converting text into numerical arrays (Bag of Words or Embeddings).

Neural Network: A multi-layer perceptron (MLP) architecture optimized for categorical cross-entropy.

📊 Evaluation
The model's performance is monitored through training_curves.png, which tracks loss and accuracy over epochs to ensure the bot generalizes well to unseen user inputs.
<img width="335" height="788" alt="Screenshot 2026-01-10 194353" src="https://github.com/user-attachments/assets/0a8d1a06-56f0-48fc-828b-361e927f3458" />
