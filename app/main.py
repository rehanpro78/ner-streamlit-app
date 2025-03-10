from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/form")
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/api/predict/", response_model=PredictionResponse)
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
