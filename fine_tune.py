import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# --- 1. Define the Multimodal Model with Attention Output ---
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
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        text_features = bert_output.pooler_output
        structured_features = self.structured_mlp(structured_data)
        combined_features = torch.cat((text_features, structured_features), dim=1)
        final_prediction = self.prediction_head(combined_features)
        return final_prediction, bert_output.attentions

# --- 2. Define the Dataset ---
class SuicideDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, scaler):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.user_text
        self.targets = dataframe.intention_score
        self.data['age_scaled'] = scaler.transform(self.data[['age']])
        self.structured = self.data[['age_scaled', 'gender_male', 'gender_female', 'gender_non_binary']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'structured_data': torch.tensor(self.structured[index], dtype=torch.float),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# --- 3. Load and Prepare Data ---
df = pd.read_csv('suicide_intention_dataset.csv')
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

age_scaler = StandardScaler()
age_scaler.fit(train_df[['age']])

MAX_LEN = 128  # Reduced for memory efficiency
TRAIN_BATCH_SIZE = 8   # Reduced for memory efficiency
VAL_BATCH_SIZE = 8
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = SuicideDataset(train_df, TOKENIZER, MAX_LEN, age_scaler)
val_dataset = SuicideDataset(val_df, TOKENIZER, MAX_LEN, age_scaler)
test_dataset = SuicideDataset(test_df, TOKENIZER, MAX_LEN, age_scaler)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# --- 4. Training and Validation Functions ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model = model.train()
    total_loss = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device, dtype=torch.long)
        attention_mask = d["attention_mask"].to(device, dtype=torch.long)
        structured_data = d["structured_data"].to(device, dtype=torch.float)
        targets = d["targets"].to(device, dtype=torch.float)
        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask, structured_data=structured_data)
        loss = loss_fn(outputs.squeeze(), targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()
    total_loss = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device, dtype=torch.long)
            attention_mask = d["attention_mask"].to(device, dtype=torch.long)
            structured_data = d["structured_data"].to(device, dtype=torch.float)
            targets = d["targets"].to(device, dtype=torch.float)
            outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask, structured_data=structured_data)
            loss = loss_fn(outputs.squeeze(), targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)

# --- 5. Fine-Tune the Model ---
EPOCHS = 10
model = MultimodalSuicideRiskModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE)
    print(f'Train loss: {train_loss:.4f}')
    val_loss = eval_model(model, val_loader, loss_fn, DEVICE)
    print(f'Validation loss: {val_loss:.4f}')
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'fine_tuned_model.bin')
        best_val_loss = val_loss

print("\nFine-tuning finished!")
print(f"Best validation loss: {best_val_loss:.4f}")