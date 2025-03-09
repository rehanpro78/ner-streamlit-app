import streamlit as st
import torch
import torch.nn as nn
from torchcrf import CRF
import joblib
import numpy as np
from typing import List, Dict

# Define the BiLSTM_CRF model class
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=100, hidden_dim=128, num_layers=2, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # CRF layer
        self.crf = CRF(tag_size, batch_first=True)
        
    def forward(self, x, mask=None, labels=None):
        # Get embeddings
        x = self.word_embeddings(x)
        x = self.dropout(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Linear layer
        emissions = self.hidden2tag(lstm_out)
        
        # If we have labels, compute loss
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            # Decode best path
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions

# Set up the Streamlit page
st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Named Entity Recognition Demo")
st.write("This app identifies named entities (like persons, organizations, locations) in text.")

# Add file uploader for model
uploaded_file = st.file_uploader("Upload your trained model file (joblib)", type=['joblib'])

if uploaded_file is not None:
    try:
        # Load the model and data
        model_data = joblib.load(uploaded_file)
        model = BiLSTM_CRF(**model_data['model_config'])
        model.load_state_dict(model_data['model_state_dict'])
        word2idx = model_data['word2idx']
        label_encoder = model_data['label_encoder']
        model.eval()
        
        # Input text box
        text_input = st.text_area("Enter your text here:", height=150)

        # Process button
        if st.button("Identify Entities"):
            if text_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                try:
                    # Tokenize input
                    tokens = text_input.split()
                    
                    # Convert tokens to ids
                    token_ids = [word2idx.get(token.lower(), word2idx['<UNK>']) 
                                for token in tokens]
                    
                    # Create tensor
                    input_ids = torch.tensor([token_ids], dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                    
                    # Get predictions
                    with torch.no_grad():
                        predictions = model(input_ids, mask=attention_mask)
                    
                    # Convert predictions to labels
                    pred_labels = label_encoder.inverse_transform(predictions[0])
                    
                    # Display results
                    st.subheader("Results:")
                    
                    # Create columns for different entity types
                    cols = st.columns(4)
                    
                    # Initialize entity lists
                    entities = {
                        'PER': [], 'ORG': [], 'LOC': [], 'MISC': []
                    }
                    
                    # Extract entities
                    current_entity = None
                    for token, label in zip(tokens, pred_labels):
                        if label.startswith('B-'):
                            if current_entity:
                                entities[current_entity['type']].append(current_entity['text'])
                            current_entity = {'text': token, 'type': label[2:]}
                        elif label.startswith('I-') and current_entity:
                            current_entity['text'] += f" {token}"
                        elif label == 'O':
                            if current_entity:
                                entities[current_entity['type']].append(current_entity['text'])
                                current_entity = None
                    
                    if current_entity:
                        entities[current_entity['type']].append(current_entity['text'])
                    
                    # Display entities in columns
                    for col, (entity_type, entity_list) in zip(cols, entities.items()):
                        with col:
                            st.write(f"**{entity_type}**")
                            if entity_list:
                                for entity in entity_list:
                                    st.write(f"- {entity}")
                            else:
                                st.write("None found")
                    
                    # Visualize tagged text
                    st.subheader("Tagged Text:")
                    html_text = []
                    for token, label in zip(tokens, pred_labels):
                        if label != 'O':
                            color = {
                                'PER': 'red',
                                'ORG': 'blue',
                                'LOC': 'green',
                                'MISC': 'purple'
                            }.get(label[2:], 'gray')
                            html_text.append(f'<span style="background-color: {color}; color: white; padding: 0.2em 0.4em; border-radius: 0.3em; margin: 0 0.2em;">{token}</span>')
                        else:
                            html_text.append(token)
                    
                    st.markdown(' '.join(html_text), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
else:
    st.info("Please upload your trained model file to begin.")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Model: BiLSTM-CRF • NER Demo")
