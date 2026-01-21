"""
Serveur Flask pour le chatbot EIDIA avec interface web
"""

from flask import Flask, render_template, request, jsonify
from hybrid_chatbot import create_hybrid_chatbot
import os

app = Flask(__name__)

# Initialiser le chatbot avec Gemini (utilise GEMINI_API_KEY du .env)
chatbot = create_hybrid_chatbot()

@app.route('/')
def index():
    """Servir la page d'accueil"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint pour traiter les messages du chatbot"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        reformulate = data.get('reformulate', True)
        
        if not user_message:
            return jsonify({'error': 'Message vide'}), 400
        
        # Obtenir la réponse du chatbot
        result = chatbot.chat(
            user_message,
            reformulate=reformulate,
            temperature=0.7,
            max_tokens=4000
        )
        
        return jsonify({
            'success': True,
            'question': result['question'],
            'intent': result['intent'],
            'confidence': f"{result['confidence']:.1%}",
            'response': result['final_response'],
            'reformulated': result['reformulated']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toggle', methods=['POST'])
def toggle():
    """Endpoint pour basculer entre les modes"""
    try:
        mode = request.json.get('mode', 'lstm')
        global chatbot
        if mode == 'lstm':
            chatbot = create_hybrid_chatbot(use_llm=False)
            return jsonify({'success': True, 'mode': 'LSTM seul'})
        else:
            chatbot = create_hybrid_chatbot()
            return jsonify({'success': True, 'mode': 'Hybride (LSTM + Gemini)'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Démarrage du serveur Flask...")
    print("Ouvrez votre navigateur à: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
