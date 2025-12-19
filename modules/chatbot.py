"""
Turkish NLP Chatbot Module
===========================
A comprehensive Turkish language chatbot using VNLP (Very Natural Language Processing)
and machine learning for intent recognition and response generation.

Author: Haidra-hub
Date: 2025-12-19
"""

import json
import pickle
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    from vnlp import NounPhraseExtractor, NamedEntityRecognizer, Lemmatizer, Tokenizer
except ImportError:
    print("Warning: VNLP library not found. Install with: pip install vnlp")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: scikit-learn library not found. Install with: pip install scikit-learn")


class TurkishNLPChatbot:
    """
    Turkish NLP Chatbot using VNLP and machine learning.
    
    Features:
    - Intent recognition using machine learning
    - Named Entity Recognition (NER)
    - Turkish text preprocessing and lemmatization
    - Contextual response generation
    - Conversation history management
    """
    
    def __init__(self, model_path: str = "models/chatbot_model.pkl"):
        """
        Initialize the Turkish NLP Chatbot.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize VNLP components
        self.tokenizer = Tokenizer()
        self.lemmatizer = Lemmatizer()
        self.ner = NamedEntityRecognizer()
        self.noun_extractor = NounPhraseExtractor()
        
        # Intent classifier pipeline
        self.intent_classifier = None
        self.tfidf_vectorizer = None
        
        # Conversation context
        self.conversation_history: List[Dict] = []
        self.context: Dict = {}
        
        # Intent templates and responses
        self.intents = self._initialize_intents()
        self.responses = self._initialize_responses()
        
        # Load model if exists
        if self.model_path.exists():
            self.load_model()
        else:
            self._train_initial_classifier()
    
    def _initialize_intents(self) -> Dict[str, List[str]]:
        """Initialize Turkish intent patterns."""
        return {
            "greeting": [
                "merhaba", "selam", "merhaba", "hoşça kalın", "hoş geldiniz",
                "nasılsın", "naaber", "naber"
            ],
            "farewell": [
                "hoşça kalın", "goodbye", "bye", "elveda", "görüşürüz",
                "sonra görüşürüz", "iyi günler"
            ],
            "help": [
                "yardım", "help", "nasıl yapılır", "nasıl çalışır", "bilgi",
                "açıkla", "anlatır mısın", "ne yapabilirim"
            ],
            "question": [
                "mi", "mi?", "mı", "mı?", "ne", "nedir", "nasıl", "kimdir",
                "ne zaman", "nerede", "niçin", "neden"
            ],
            "affirmation": [
                "evet", "yes", "tamam", "pekala", "haklısın", "doğru",
                "kabul ediyorum", "anlaştık"
            ],
            "negation": [
                "hayır", "no", "yok", "değil", "istemiyorum", "olmaz",
                "kabul edemem", "reddediyorum"
            ],
            "thanks": [
                "teşekkür", "thanks", "sağol", "teşekkür ederim", "çok sağol",
                "merci", "mamnun oldum"
            ]
        }
    
    def _initialize_responses(self) -> Dict[str, List[str]]:
        """Initialize response templates for different intents."""
        return {
            "greeting": [
                "Merhaba! Sizi nasıl yardımcı olabilirim?",
                "Hoş geldiniz! Ne yapabilirim sizin için?",
                "Selam! Bugün ne istiyorsunuz?",
                "Merhaba! Nasılsınız?"
            ],
            "farewell": [
                "Hoşça kalın! Tekrar görüşmek üzere!",
                "Sonra görüşmek üzere! İyi günler!",
                "Elveda! Başarılar dilerim!",
                "Görüşürüz! Kendine iyi bak!"
            ],
            "help": [
                "Size yardımcı olmaktan mutluluk duyarım. Ne istiyorsunuz?",
                "Tabii! Lütfen sorunuzu detaylı şekilde anlatın.",
                "Elbette yardımcı olabilirim. Nasıl başlayalım?",
                "Kimseye yardım etmek benim görevim. Neye ihtiyacınız var?"
            ],
            "affirmation": [
                "Harika! Devam edelim.",
                "Anlaştık! Sonraki adım nedir?",
                "Mükemmel! Başka ne yapabilirim?",
                "Tamam! Başarıyla tamamlandı."
            ],
            "negation": [
                "Anladım. Başka bir şey deneyebiliriz.",
                "Sorun değil. Farklı bir yaklaşım deneyelim.",
                "Tamam, başka bir seçenek var mı?",
                "Merak etmeyin, farklı bir yol buluruz."
            ],
            "thanks": [
                "Rica ederim! Başka bir şey yardımcı olabilirim mi?",
                "Memnun oldum! Herhangi başka bir ihtiyacınız var mı?",
                "Sizin için yardımcı olmak bir zevk! Başka ne gerek?",
                "Çok üzgün değilim! Başka ne yapabilirim?"
            ],
            "default": [
                "Anladım. Bu konuda daha fazla bilgi alabilir misiniz?",
                "İlginç bir soru. Bunu açıklamanıza yardımcı olabilirim.",
                "Belki daha detaylı bilgi verebilirsiniz?",
                "Anladım. Başka ne merak ediyorsunuz?"
            ]
        }
    
    def preprocess_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Preprocess Turkish text.
        
        Args:
            text: Raw input text
            
        Returns:
            Tuple of (normalized_text, tokens)
        """
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove special characters but keep Turkish characters
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0400-\u04FF]', '', text)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(lemmatized_tokens), lemmatized_tokens
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        try:
            entities = self.ner.predict(text)
            return entities if entities else {}
        except Exception as e:
            print(f"Error in NER: {e}")
            return {}
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text.
        
        Args:
            text: Input text
            
        Returns:
            List of noun phrases
        """
        try:
            noun_phrases = self.noun_extractor.extract(text)
            return noun_phrases if noun_phrases else []
        except Exception as e:
            print(f"Error in noun phrase extraction: {e}")
            return []
    
    def recognize_intent(self, text: str) -> Tuple[str, float]:
        """
        Recognize user intent from text.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (intent, confidence)
        """
        preprocessed_text, tokens = self.preprocess_text(text)
        
        # Check against intent patterns
        max_score = 0
        detected_intent = "default"
        
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if pattern in preprocessed_text or pattern in tokens:
                    score = 1.0
                    if score > max_score:
                        max_score = score
                        detected_intent = intent
        
        # Use ML classifier if trained
        if self.intent_classifier is not None:
            try:
                probabilities = self.intent_classifier.predict_proba(
                    [preprocessed_text]
                )[0]
                ml_intent = self.intent_classifier.classes_[np.argmax(probabilities)]
                ml_confidence = np.max(probabilities)
                
                if ml_confidence > max_score:
                    detected_intent = ml_intent
                    max_score = ml_confidence
            except Exception as e:
                print(f"Error in ML classification: {e}")
        
        return detected_intent, min(max_score, 0.99)
    
    def _train_initial_classifier(self):
        """Train initial intent classifier with sample data."""
        try:
            # Prepare training data
            training_texts = []
            training_labels = []
            
            for intent, patterns in self.intents.items():
                for pattern in patterns:
                    training_texts.append(pattern)
                    training_labels.append(intent)
            
            # Create and train pipeline
            self.intent_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100, lowercase=True)),
                ('nb', MultinomialNB())
            ])
            
            self.intent_classifier.fit(training_texts, training_labels)
        except Exception as e:
            print(f"Error training classifier: {e}")
    
    def generate_response(self, intent: str, context: Optional[Dict] = None) -> str:
        """
        Generate response based on intent.
        
        Args:
            intent: Detected intent
            context: Optional context information
            
        Returns:
            Generated response string
        """
        intent_responses = self.responses.get(intent, self.responses["default"])
        
        if intent_responses:
            # Select response based on context or randomly
            response = intent_responses[hash(datetime.now().isoformat()) % len(intent_responses)]
            return response
        
        return "Anladım. Nasıl yardımcı olabilirim?"
    
    def chat(self, user_input: str) -> str:
        """
        Main chat method. Process user input and generate response.
        
        Args:
            user_input: User's message
            
        Returns:
            Chatbot's response
        """
        # Store in history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "user",
            "message": user_input
        })
        
        # Recognize intent
        intent, confidence = self.recognize_intent(user_input)
        
        # Extract entities
        entities = self.extract_entities(user_input)
        noun_phrases = self.extract_noun_phrases(user_input)
        
        # Update context
        self.context["last_intent"] = intent
        self.context["last_confidence"] = confidence
        self.context["last_entities"] = entities
        self.context["last_noun_phrases"] = noun_phrases
        
        # Generate response
        response = self.generate_response(intent, self.context)
        
        # Store response in history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "bot",
            "message": response,
            "intent": intent,
            "confidence": confidence
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.context = {}
    
    def save_model(self):
        """Save trained model to disk."""
        try:
            model_data = {
                "intent_classifier": self.intent_classifier,
                "intents": self.intents,
                "responses": self.responses
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load trained model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.intent_classifier = model_data.get("intent_classifier")
            self.intents = model_data.get("intents", self.intents)
            self.responses = model_data.get("responses", self.responses)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def get_context(self) -> Dict:
        """Get current conversation context."""
        return self.context.copy()
    
    def get_statistics(self) -> Dict:
        """Get conversation statistics."""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for m in self.conversation_history if m["speaker"] == "user"),
            "bot_messages": sum(1 for m in self.conversation_history if m["speaker"] == "bot"),
            "intents_detected": list(set(
                m.get("intent") for m in self.conversation_history 
                if m.get("intent")
            )),
            "average_confidence": np.mean([
                m.get("confidence", 0) for m in self.conversation_history 
                if m.get("confidence")
            ]) if self.conversation_history else 0
        }


class TurkishChatbotTrainer:
    """
    Trainer class for improving chatbot performance.
    """
    
    def __init__(self, chatbot: TurkishNLPChatbot):
        """
        Initialize trainer.
        
        Args:
            chatbot: TurkishNLPChatbot instance to train
        """
        self.chatbot = chatbot
        self.training_data: List[Tuple[str, str]] = []
    
    def add_training_example(self, text: str, intent: str):
        """
        Add training example.
        
        Args:
            text: Example text
            intent: Correct intent label
        """
        self.training_data.append((text, intent))
    
    def train(self):
        """Train the classifier with collected examples."""
        if not self.training_data:
            print("No training data provided.")
            return
        
        texts, intents = zip(*self.training_data)
        
        try:
            self.chatbot.intent_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100, lowercase=True)),
                ('nb', MultinomialNB())
            ])
            self.chatbot.intent_classifier.fit(texts, intents)
            self.chatbot.save_model()
            print(f"Training completed with {len(self.training_data)} examples.")
        except Exception as e:
            print(f"Error during training: {e}")
    
    def clear_training_data(self):
        """Clear training data."""
        self.training_data = []


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = TurkishNLPChatbot()
    
    # Example conversations
    test_inputs = [
        "Merhaba! Nasılsın?",
        "Ne yapabilirim?",
        "Teşekkür ederim!",
        "Hoşça kalın!"
    ]
    
    print("Turkish NLP Chatbot Test")
    print("=" * 50)
    
    for user_input in test_inputs:
        response = chatbot.chat(user_input)
        print(f"\nUser: {user_input}")
        print(f"Bot: {response}")
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Conversation Statistics:")
    stats = chatbot.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
