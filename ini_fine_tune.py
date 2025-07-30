import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import joblib

# --- 1. Define Model and Dataset Classes ---
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

    def forward(self, input_ids, attention_mask, structured_data):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output
        structured_features = self.structured_mlp(structured_data)
        combined_features = torch.cat((text_features, structured_features), dim=1)
        return self.prediction_head(combined_features)

class SuicideDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, scaler):
        self.tokenizer, self.data, self.max_len = tokenizer, dataframe.copy(), max_len
        self.text, self.targets = self.data.user_text, self.data.intention_score
        self.data['age_scaled'] = scaler.transform(self.data[['age']])
        self.structured = self.data[['age_scaled', 'gender_male', 'gender_female', 'gender_non_binary']].values

    def __len__(self): return len(self.text)
    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=self.max_len, padding='max_length',
            return_token_type_ids=False, truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(), 'attention_mask': inputs['attention_mask'].flatten(),
            'structured_data': torch.tensor(self.structured[index], dtype=torch.float),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# --- 2. Define Training and Evaluation Functions ---
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for d in data_loader:
        input_ids, mask, struct, targets = d['input_ids'].to(device), d['attention_mask'].to(device), d['structured_data'].to(device), d['targets'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=mask, structured_data=struct)
        loss = loss_fn(outputs.squeeze(), targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids, mask, struct, targets = d['input_ids'].to(device), d['attention_mask'].to(device), d['structured_data'].to(device), d['targets'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask, structured_data=struct)
            loss = loss_fn(outputs.squeeze(), targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    df = pd.read_csv('suicide_intention_dataset.csv')
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    age_scaler = StandardScaler().fit(train_df[['age']])
    # Save the scaler with the .bin extension
    joblib.dump(age_scaler, 'age_scaler.bin')
    print("Scaler fitted and saved as 'age_scaler.bin'")


    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = SuicideDataset(train_df, TOKENIZER, 128, age_scaler)
    val_dataset = SuicideDataset(val_df, TOKENIZER, 128, age_scaler)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = MultimodalSuicideRiskModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')

    print("\n--- Starting Initial Fine-Tuning ---")
    for epoch in range(10): # 10 epochs for initial training
        print(f'Epoch {epoch + 1}/10')
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
        val_loss = eval_model(model, val_loader, loss_fn, DEVICE)
        print(f'  Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'fine_tuned_model.bin')
            print(f"  New best model saved to 'fine_tuned_model.bin'")
    print("\nInitial fine-tuning complete.")
