# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import joblib
import os

# -----------------------
# Config
# -----------------------
MODEL_WEIGHTS_PATH = "model_weights.h5"
MAX_LEN = 128

ENCODER_PATHS = {
    "type": "le_type.pkl",
    "rel": "le_rel.pkl",
    "urg": "le_urg.pkl"
}

# -----------------------
# Load Label Encoders
# -----------------------
def load_encoder(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing encoder: {path}")
    return joblib.load(path)

le_type = load_encoder(ENCODER_PATHS["type"])
le_rel  = load_encoder(ENCODER_PATHS["rel"])
le_urg  = load_encoder(ENCODER_PATHS["urg"])

NUM_TYPE = len(le_type.classes_)
NUM_REL  = len(le_rel.classes_)
NUM_URG  = len(le_urg.classes_)

# -----------------------
# Load tokenizer & model
# -----------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
backbone = TFAutoModel.from_pretrained("distilbert-base-uncased", from_pt=True)

input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

outputs = backbone(input_ids, attention_mask=attention_mask)
cls_token = outputs.last_hidden_state[:, 0, :]
x = tf.keras.layers.Dropout(0.2)(cls_token)

type_output = tf.keras.layers.Dense(NUM_TYPE, activation="softmax", name="type_output")(x)
rel_output  = tf.keras.layers.Dense(NUM_REL, activation="softmax", name="rel_output")(x)
urg_output  = tf.keras.layers.Dense(NUM_URG, activation="softmax", name="urg_output")(x)

model = tf.keras.Model(inputs=[input_ids, attention_mask],
                       outputs=[type_output, rel_output, urg_output])

model.load_weights(MODEL_WEIGHTS_PATH)

# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="Disaster MultiTask API")

class InputText(BaseModel):
    text: str

DISASTER_KEYWORDS = [
    "flood", "hurricane", "cyclone", "storm", "tsunami",
    "high wave", "tidal", "surge", "landslide", "rain",
    "evacuate", "warning", "alert", "emergency"
]

@app.post("/predict")
def predict(data: InputText):
    text_lower = data.text.lower()
    if not any(word in text_lower for word in DISASTER_KEYWORDS):
        return {
            "text": data.text,
            "results": {
                "disaster_type": "casual",
                "relevance": "0",
                "urgency": "neutral"
            }
        }

    tokens = tokenizer(
        data.text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="tf"
    )

    outputs = model([tokens["input_ids"], tokens["attention_mask"]])

    type_pred = le_type.inverse_transform([int(tf.argmax(outputs[0], axis=-1))])[0]
    rel_pred  = le_rel.inverse_transform([int(tf.argmax(outputs[1], axis=-1))])[0]
    urg_pred  = le_urg.inverse_transform([int(tf.argmax(outputs[2], axis=-1))])[0]

    return {
        "text": data.text,
        "results": {
            "disaster_type": str(type_pred),
            "relevance": str(rel_pred),
            "urgency": str(urg_pred)
        }
    }
