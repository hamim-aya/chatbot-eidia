document.addEventListener('DOMContentLoaded', function() {
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const messagesContainer = document.getElementById('messages');
    const reformulateCheckbox = document.getElementById('reformulate');

    // Afficher l'heure du message de bienvenue
    const welcomeTime = document.getElementById('welcome-time');
    if (welcomeTime) {
        welcomeTime.textContent = new Date().toLocaleTimeString('fr-FR', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    // Envoyer un message
    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Ajouter le message de l'utilisateur
        addMessage(message, 'user');
        userInput.value = '';
        sendBtn.disabled = true;

        // Afficher l'indicateur de chargement
        addLoadingIndicator();

        // Envoyer au serveur
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                reformulate: reformulateCheckbox.checked
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Erreur serveur');
            }
            return response.json();
        })
        .then(data => {
            removeLoadingIndicator();
            
            if (data.success) {
                // Construire la réponse
                let response = data.response;
                
                // Ajouter le message du bot
                addMessage(response, 'bot', {
                    intent: data.intent,
                    confidence: data.confidence,
                    reformulated: data.reformulated
                });
            } else {
                addMessage('Erreur: ' + (data.error || 'Réponse vide'), 'error');
            }
        })
        .catch(error => {
            removeLoadingIndicator();
            console.error('Erreur:', error);
            addMessage('❌ Erreur de connexion: ' + error.message, 'error');
        })
        .finally(() => {
            sendBtn.disabled = false;
            userInput.focus();
        });
    }

    // Ajouter un message au chat
    function addMessage(text, type, metadata = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        // Contenu du message
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Gérer les sauts de ligne et forcer le word-wrap
        const paragraphs = text.split('\n').filter(line => line.trim());
        if (paragraphs.length === 0 && text.trim()) {
            // Si pas de sauts de ligne, afficher le texte complet
            const p = document.createElement('p');
            p.textContent = text;
            p.style.wordWrap = 'break-word';
            p.style.overflowWrap = 'break-word';
            p.style.whiteSpace = 'normal';
            contentDiv.appendChild(p);
        } else {
            // Sinon, diviser par paragraphes
            paragraphs.forEach(paragraph => {
                const p = document.createElement('p');
                p.textContent = paragraph;
                p.style.wordWrap = 'break-word';
                p.style.overflowWrap = 'break-word';
                p.style.whiteSpace = 'normal';
                contentDiv.appendChild(p);
            });
        }

        messageDiv.appendChild(contentDiv);

        // Métadonnées
        if (metadata || type === 'bot') {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';

            // Timestamp
            const timeSpan = document.createElement('span');
            timeSpan.className = 'timestamp';
            timeSpan.textContent = new Date().toLocaleTimeString('fr-FR', {
                hour: '2-digit',
                minute: '2-digit'
            });
            metaDiv.appendChild(timeSpan);

            // Intent et confidence
            if (metadata && metadata.intent) {
                const intentBadge = document.createElement('span');
                intentBadge.className = 'intent-badge';
                intentBadge.textContent = `🎯 ${metadata.intent}`;
                metaDiv.appendChild(intentBadge);

                const confidenceBadge = document.createElement('span');
                confidenceBadge.className = 'confidence';
                confidenceBadge.textContent = `${metadata.confidence}`;
                metaDiv.appendChild(confidenceBadge);
            }

            // Indicateur de reformulation
            if (metadata && metadata.reformulated) {
                const reformulatedBadge = document.createElement('span');
                reformulatedBadge.className = 'reformulated-badge';
                reformulatedBadge.textContent = '✨ Reformulée';
                metaDiv.appendChild(reformulatedBadge);
            }

            messageDiv.appendChild(metaDiv);
        }

        messagesContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    // Ajouter l'indicateur de chargement
    function addLoadingIndicator() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading';
        loadingDiv.id = 'loading-indicator';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'loading-dot';
            loadingDiv.appendChild(dot);
        }

        messagesContainer.appendChild(loadingDiv);
        scrollToBottom();
    }

    // Supprimer l'indicateur de chargement
    function removeLoadingIndicator() {
        const loadingDiv = document.getElementById('loading-indicator');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    // Scroller vers le bas
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Événements
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    // Bouton effacer
    clearBtn.addEventListener('click', function() {
        if (confirm('Êtes-vous sûr de vouloir effacer le chat?')) {
            messagesContainer.innerHTML = `
                <div class="message bot-message">
                    <div class="message-content">
                        <p>Bienvenue! 👋 Je suis l'assistant EIDIA-UEMF.</p>
                        <p>Comment puis-je vous aider?</p>
                    </div>
                    <div class="message-meta">
                        <span class="timestamp">${new Date().toLocaleTimeString('fr-FR', {
                            hour: '2-digit',
                            minute: '2-digit'
                        })}</span>
                    </div>
                </div>
            `;
            userInput.focus();
        }
    });

    // Focus initial
    userInput.focus();
});
