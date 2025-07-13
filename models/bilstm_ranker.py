# models/bilstm_ranker.py
import torch
import torch.nn as nn

class BiLSTMRanker(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMRanker, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

    def training_step(self, batch, optimizer, criterion):
        self.train()
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()
        outputs = self.forward(inputs)
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate_step(self, batch):
        self.eval()
        with torch.no_grad():
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.forward(inputs)
            predictions = torch.argmax(outputs, dim=-1)
        return predictions.cpu(), labels.cpu()

    def set_device(self, device):
        self.device = device
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
