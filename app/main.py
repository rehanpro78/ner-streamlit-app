from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import joblib
from typing import List, Dict
import os
import sys

# Dynamically import the model 
# Assuming you have a model.py in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model import BiLSTM_CRF
except ImportError:
    # Fallback or define a simple placeholder model class
    class BiLSTM_CRF:
        def __init__(self, **kwargs):
            pass
        def load_state_dict(self, state_dict):
            pass
        def eval(self):
            pass

app = FastAPI(
    title="NER API",
    description="API for Named Entity Recognition",
    version="1.0.0"
)

# Modify static and template mounting
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, 'static')
templates_dir = os.path.join(base_dir, 'templates')

# Create directories if they don't exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Model loading with more robust error handling
model = None
word2idx = None
label_encoder = None

def load_model():
    global model, word2idx, label_encoder
    try:
        model_path = os.path.join(base_dir, 'models', 'ner_model.joblib')
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        model_data = joblib.load(model_path)
        model = BiLSTM_CRF(**model_data['model_config'])
        model.load_state_dict(model_data['model_state_dict'])
        word2idx = model_data['word2idx']
        label_encoder = model_data['label_encoder']
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Try to load model on startup
model_loaded = load_model()

#change update 0.02
# Pydantic Models for Request/Response
class TextRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int

class PredictionResponse(BaseModel):
    entities: List[Entity]
    text: str


#difining endpoints, start with root
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#adding update remaining 0.03
@app.get("/uipredict")
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/", response_model=PredictionResponse)
async def predict(request: TextRequest):
    global model, word2idx, label_encoder
    
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Tokenize input
        tokens = request.text.split()
        
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
        pred_labels = [label_encoder.inverse_transform([p])[0] for p in predictions[0]]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if isinstance(label, str) and label.startswith('B-'):
                if current_entity:
                    entities.append(Entity(**current_entity))
                current_entity = {
                    'text': token,
                    'label': label[2:],
                    'start': i,
                    'end': i + 1
                }
            elif isinstance(label, str) and label.startswith('I-') and current_entity:
                current_entity['text'] += f" {token}"
                current_entity['end'] = i + 1
            elif label == 'O':
                if current_entity:
                    entities.append(Entity(**current_entity))
                    current_entity = None
        
        if current_entity:
            entities.append(Entity(**current_entity))
        
        return PredictionResponse(
            entities=entities,
            text=request.text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def process_prediction(request: Request):
    form = await request.form()
    text = form.get('text', '')
    
    try:
        # Call the predict function
        prediction = await predict(TextRequest(text=text))
        
        # Render output template with prediction results
        return templates.TemplateResponse("output.html", {
            "request": request, 
            "text": text, 
            "entities": prediction.entities
        })
    except Exception as e:
        return templates.TemplateResponse("output.html", {
            "request": request, 
            "text": text, 
            "error": str(e)
        })


# Add a health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
from typing import List, Dict
from .model import BiLSTM_CRF
import numpy as np

app = FastAPI(
    title="NER API",
    description="API for Named Entity Recognition",
    version="1.0.0"
)

# Load model and components
try:
    model_data = joblib.load('app/models/ner_model.joblib')
    model = BiLSTM_CRF(**model_data['model_config'])
    model.load_state_dict(model_data['model_state_dict'])
    word2idx = model_data['word2idx']
    label_encoder = model_data['label_encoder']
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")

class TextRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int

class PredictionResponse(BaseModel):
    entities: List[Entity]
    text: str

@app.get("/")
def read_root():
    return {"message": "NER API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        # Tokenize input
        tokens = request.text.split()
        
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
        # Convert the predictions to a numpy array first
        pred_labels = [label_encoder.inverse_transform([p])[0] for p in predictions[0]]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if isinstance(label, str) and label.startswith('B-'):
                if current_entity:
                    entities.append(Entity(**current_entity))
                current_entity = {
                    'text': token,
                    'label': label[2:],
                    'start': i,
                    'end': i + 1
                }
            elif isinstance(label, str) and label.startswith('I-') and current_entity:
                current_entity['text'] += f" {token}"
                current_entity['end'] = i + 1
            elif label == 'O':
                if current_entity:
                    entities.append(Entity(**current_entity))
                    current_entity = None
        
        if current_entity:
            entities.append(Entity(**current_entity))
        
        return PredictionResponse(
            entities=entities,
            text=request.text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''







'''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
from typing import List, Dict
from .model import BiLSTM_CRF

app = FastAPI(
    title="NER API",
    description="API for Named Entity Recognition",
    version="1.0.0"
)

# Load model and components
try:
    model_data = joblib.load('app/models/ner_model.joblib')
    model = BiLSTM_CRF(**model_data['model_config'])
    model.load_state_dict(model_data['model_state_dict'])
    word2idx = model_data['word2idx']
    label_encoder = model_data['label_encoder']
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")

class TextRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int

class PredictionResponse(BaseModel):
    entities: List[Entity]
    text: str

@app.get("/")
def read_root():
    return {"message": "NER API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        # Tokenize input
        tokens = request.text.split()
        
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
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(Entity(**current_entity))
                current_entity = {
                    'text': token,
                    'label': label[2:],
                    'start': i,
                    'end': i + 1
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += f" {token}"
                current_entity['end'] = i + 1
            elif label == 'O':
                if current_entity:
                    entities.append(Entity(**current_entity))
                    current_entity = None
        
        if current_entity:
            entities.append(Entity(**current_entity))
        
        return PredictionResponse(
            entities=entities,
            text=request.text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''
