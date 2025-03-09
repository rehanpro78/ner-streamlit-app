import streamlit as st
import torch
import joblib
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Load the model and necessary components
@st.cache_resource
def load_model():
    model_data = joblib.load('model/ner_model.joblib')
    model = BiLSTM_CRF(**model_data['model_config'])
    model.load_state_dict(model_data['model_state_dict'])
    return model, model_data

# Define the BiLSTM_CRF class (copy your model class here)
class BiLSTM_CRF(nn.Module):
    # Copy your model class implementation here
    # (The same code we used in Colab)
    pass

# Set up the Streamlit page
st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Named Entity Recognition Demo")
st.write("This app identifies named entities (like persons, organizations, locations) in text.")

# Load the model
try:
    model, model_data = load_model()
    word2idx = model_data['word2idx']
    label_encoder = model_data['label_encoder']
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

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

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Model: BiLSTM-CRF • [GitHub Repository](your-repo-link)")
