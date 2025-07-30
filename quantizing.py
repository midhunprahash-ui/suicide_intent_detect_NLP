import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import joblib
from bertviz import head_view
import copy

# --- 1. Define Model Class ---
# Must be identical to the one used during training.
class MultimodalSuicideRiskModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', n_structured_features=4):
        super(MultimodalSuicideRiskModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_output_size = self.bert.config.hidden_size
        self.structured_mlp = nn.Sequential(
            nn.Linear(n_structured_features, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2)
        )
        self.mlp_output_size = 16
        combined_feature_size = self.bert_output_size + self.mlp_output_size
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_feature_size, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, structured_data, output_attentions=False):
        if output_attentions:
            self.bert.config.output_attentions = True
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        else:
            self.bert.config.output_attentions = False
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
        text_features = bert_output.pooler_output
        structured_features = self.structured_mlp(structured_data)
        combined_features = torch.cat((text_features, structured_features), dim=1)
        final_prediction = self.prediction_head(combined_features)
        
        if output_attentions:
            return final_prediction, bert_output.attentions
        return final_prediction

# --- 2. Quantization and Inference ---
if __name__ == '__main__':
    DEVICE = "cpu" # Quantization is for CPU inference
    print(f"Using device: {DEVICE}")

    print("\n--- Loading pruned and fine-tuned model for quantization ---")
    model = MultimodalSuicideRiskModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load('pruned_and_finetuned_model.bin', map_location=DEVICE))
    except FileNotFoundError:
        print("\nERROR: 'pruned_and_finetuned_model.bin' not found. Run script #3 first.")
        exit()
    model.eval()
    
    # Keep a copy for visualization before it gets quantized
    viz_model = copy.deepcopy(model)

    print("\n--- Applying Dynamic Quantization ---")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), 'final_quantized_model.bin')
    print("Quantization complete. Final model saved as 'final_quantized_model.bin'")

    # --- 3. Inference ---
    print("\n--- Loading components for Inference ---")
    try:
        TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load the scaler with the correct .bin extension
        SCALER = joblib.load('age_scaler.bin')
    except FileNotFoundError:
        print("\nERROR: 'age_scaler.bin' not found. Run script #1 to create it.")
        exit()

    # --- Example Prediction ---
    text = "I feel so alone and don’t see a way out of this pain."
    age = 22
    gender = "male" # 'male', 'female', or 'other'

    print(f"\n--- Running Inference for: '{text}' ---")
    gender_map = {'male': [1,0,0], 'female': [0,1,0], 'other': [0,0,1]}
    gender_encoding = gender_map.get(gender.lower(), [0,0,1])
    scaled_age = SCALER.transform([[age]])[0][0]
    structured_list = [scaled_age] + gender_encoding
    structured_data = torch.tensor([structured_list], dtype=torch.float).to(DEVICE)
    
    inputs = TOKENIZER(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        prediction = quantized_model(input_ids, attention_mask, structured_data)
    score = prediction.item()
    print(f"Predicted Intensity Score: {score:.2f}/10")
    
    if score >= 7:
        print("\n" + "*"*65)
        print("! High-risk prediction. This requires immediate attention.")
        print("! Contact: 1-800-273-8255 (USA) or your local emergency helpline.")
        print("*"*65)
        
    with torch.no_grad():
        _, attentions = viz_model(input_ids, attention_mask, structured_data, output_attentions=True)
        
    print("\nDisplaying Attention Visualization...")
    tokens = TOKENIZER.convert_ids_to_tokens(input_ids[0])
    head_view(attentions, tokens)
