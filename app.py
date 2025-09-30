import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import emoji
from transformers import AutoTokenizer, AutoModel

# -------------------
# Pydantic model
# -------------------
class Tweet(BaseModel):
    text: str

# -------------------
# Load tokenizer & BERT
# -------------------
model_name = "bert-base-uncased"  
bert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.save_pretrained("model/bert-tokenizer")  # optional local save

# -------------------
# Multi-task heads
# -------------------
num_disaster = 7
num_urgency  = 3

class MultiTaskBERT(torch.nn.Module):
    def __init__(self, bert, num_disaster, num_urgency):
        super().__init__()
        self.bert = bert
        hidden_size = self.bert.config.hidden_size
        self.disaster_head = torch.nn.Linear(hidden_size, num_disaster)
        self.urgency_head   = torch.nn.Linear(hidden_size, num_urgency)
        self.relevance_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return (self.disaster_head(pooled),
                self.urgency_head(pooled),
                self.relevance_head(pooled))

# -------------------
# Initialize model
# -------------------
model = MultiTaskBERT(bert, num_disaster, num_urgency)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------
# Load checkpoint safely
# -------------------
heads = torch.load("multi_task_bert.pth", map_location=device)
print("Checkpoint keys:", heads.keys())

for head_name in ["disaster_head", "urgency_head", "relevance_head"]:
    if f"{head_name}.weight" in heads and f"{head_name}.bias" in heads:
        getattr(model, head_name).weight.data = heads[f"{head_name}.weight"]
        getattr(model, head_name).bias.data   = heads[f"{head_name}.bias"]
    else:
        print(f"Warning: {head_name} not found in checkpoint!")

model.eval()

# -------------------
# Label mappings
# -------------------
disaster_labels = ['flood','hurricane','rain','cyclone','storm','high_waves','casual']
urgency_labels  = ['neutral','panic','emergency']

# -------------------
# FastAPI app
# -------------------
app = FastAPI(title="Disaster Tweet Classifier")

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return emoji.demojize(text, language="en")

def predict_tweet(text: str) -> dict:
    inputs = tokenizer(preprocess_text(text), padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        d_logits, u_logits, r_logits = model(inputs['input_ids'], inputs['attention_mask'])
        disaster_idx = torch.argmax(d_logits, dim=1).item()
        urgency_idx  = torch.argmax(u_logits, dim=1).item()
        relevance    = torch.sigmoid(r_logits).item()
    return {
        "disaster_type": disaster_labels[disaster_idx],
        "urgency": urgency_labels[urgency_idx],
        "relevance": round(relevance, 3)
    }

@app.post("/predict")
def predict(tweet: Tweet):
    return predict_tweet(tweet.text)

# -------------------
# Run server
# -------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
