import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

## ----------------- Step 1: Define the Model Architecture -----------------
# This must be the exact same class definition you used for training.

class MultimodalSuicideRiskModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', n_structured_features=4):
        super(MultimodalSuicideRiskModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_output_size = self.bert.config.hidden_size
        self.structured_mlp = nn.Sequential(
            nn.Linear(n_structured_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.mlp_output_size = 16
        combined_feature_size = self.bert_output_size + self.mlp_output_size
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, structured_data):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output
        structured_features = self.structured_mlp(structured_data)
        combined_features = torch.cat((text_features, structured_features), dim=1)
        final_prediction = self.prediction_head(combined_features)
        return final_prediction

## -------------------- Step 2: Load All Necessary Assets --------------------

def load_assets(model_path, scaler_path):
    """Loads all necessary files and objects for inference."""
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Missing required files!")
        print(f"Please make sure '{model_path}' and '{scaler_path}' are in the same directory.")
        return None, None, None, None, None
        
    # Define constants
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 160

    # Load the fitted scaler
    age_scaler = joblib.load(scaler_path)
    mean_age = age_scaler.mean_[0]

    # Instantiate and load the model
    model = MultimodalSuicideRiskModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    
    print("Model and assets loaded successfully.")
    return model, tokenizer, age_scaler, mean_age, device, max_len

## ----------------- Step 3: Define the Universal Prediction Function -----------------

def predict_intention(model, tokenizer, age_scaler, mean_age, device, max_len, text, age=None, gender_str=None):
    """Predicts the intention score using the loaded model and assets."""
    final_age = age if age is not None else mean_age
    
    gender_map = {'male': [1,0,0], 'female': [0,1,0], 'non_binary': [0,0,1]}
    gender_encoding = gender_map.get(gender_str.lower(), [1/3, 1/3, 1/3]) if gender_str else [1/3, 1/3, 1/3]

    scaled_age = age_scaler.transform([[final_age]])[0]
    structured_data_np = np.concatenate([scaled_age, gender_encoding])
    
    encoded_text = tokenizer.encode_plus(
        text, max_length=max_len, add_special_tokens=True, return_token_type_ids=False,
        padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True
    )
    
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    structured_tensor = torch.tensor(structured_data_np, dtype=torch.float).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            structured_data=structured_tensor
        )
    return prediction.item()

## ------------------------- Step 4: Create the Interactive Loop -------------------------

def run_interactive_session():
    # Load all assets first
    model, tokenizer, age_scaler, mean_age, device, max_len = load_assets(
        model_path='/Users/midhun/Developer/Git/PyNDA/MODELS/best_model_state_2.bin',
        scaler_path='/Users/midhun/Developer/Git/PyNDA/MODELS/age_scaler_2.bin'
    )
    
    # If loading failed, exit the script
    if model is None:
        return
        
    print("\n--- Suicide Intention Predictor ---")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        # --- Get User Input ---
        text_input = input("Enter the user text (mandatory): ")
        if text_input.lower() in ['quit', 'exit']:
            print("Session ended. Goodbye!")
            break

        age_input = input("Enter age (optional, press Enter to skip): ")
        gender_input = input("Enter gender [male/female/non_binary] (optional, press Enter to skip): ")
        
        # --- Process Inputs ---
        try:
            # Convert age to int if provided, otherwise it's None
            age = int(age_input) if age_input else None
            # Use gender string if provided, otherwise it's None
            gender = gender_input if gender_input else None
            
            # --- Make Prediction ---
            score = predict_intention(model, tokenizer, age_scaler, mean_age, device, max_len,
                                      text=text_input, age=age, gender_str=gender)
            
            # --- Display Result ---
            print("\n---------------------------------")
            print(f"Predicted Intention Score: {score:.2f} / 10.0")
            print("---------------------------------\n")

        except ValueError:
            print("\nInvalid age. Please enter a number for the age. Try again.\n")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}. Please try again.\n")

# This ensures the script runs only when executed directly
if __name__ == "__main__":
    run_interactive_session()