import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import BertModel

# --- 1. Define Model Class ---
# Must be identical to the one used during training to load the state_dict.
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

# --- 2. Main Execution Block ---
if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    print("\n--- Loading the fine-tuned model for pruning ---")
    model = MultimodalSuicideRiskModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load('fine_tuned_model.bin', map_location=DEVICE))
    except FileNotFoundError:
        print("\nERROR: 'fine_tuned_model.bin' not found. Run script #1 first.")
        exit()

    print("\n--- Applying Pruning ---")
    # Unstructured weight pruning
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight') # Make pruning permanent

    # Structured head pruning
    heads_to_prune = {10: [0, 1], 11: [2, 3]} # Prune from top layers
    model.bert.config.output_attentions = True
    for layer, heads in heads_to_prune.items():
        model.bert.encoder.layer[layer].attention.prune_heads(heads)
    
    torch.save(model.state_dict(), 'pruned_only_model.bin')
    print("Pruning complete. Pruned model saved as 'pruned_only_model.bin'")
