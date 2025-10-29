import torch                        #PyTorch
import torch.nn as nn               #Camadas da Rede neural
import torch.optim as optim         #Otimizadores
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from scipy.signal import firwin, filtfilt
import copy

def apply_fir_bandpass(signal, fs=160.0, lowcut=30.0, highcut=50.0, order=12):
    """
    Aplica filtro FIR passa-banda (igual ao código TensorFlow)
    signal: np.ndarray (C, T) ou (T, C)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # FIR filter usando firwin (igual ao TensorFlow)
    fir_coeff = firwin(order+1, [low, high], pass_zero=False)
    
    # Aplica filtro no último eixo
    filtered_signal = filtfilt(fir_coeff, 1.0, signal, axis=-1)
    
    return np.ascontiguousarray(filtered_signal)

def normalize_sun(signal):
    """
    Normalização 'sun' (igual ao código TensorFlow)
    signal: np.ndarray (T, C) - forma esperada após a transposição
    """
    # Normalização por canal (cada coluna)
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    return (signal - mean) / (std + 1e-8)

class EEGWindowDataset(Dataset):
    def __init__(self, folder_path, subjects, tasks, sampling_points=1920, offset=35, label_map=None, transform=None):
        self.windows = []
        self.labels = []
        self.transform = transform
        self.fs = 160.0

        for subj in subjects:
            for task in tasks:
                file_name = f"S{subj:03d}{task}.csv"
                file_path = os.path.join(folder_path, f"S{subj:03d}", file_name)

                if not os.path.exists(file_path):
                    print(f"[AVISO] Arquivo não encontrado: {file_path}")
                    continue

                data = np.loadtxt(file_path, delimiter=',')  # shape: (64, total_amostras)
                # Aplicar filtro uma única vez por arquivo
                filtered_data = apply_fir_bandpass(data, fs=self.fs, lowcut=30.0, highcut=50.0, order=12)
                
                total_points = filtered_data.shape[1]

                for start in range(0, total_points - sampling_points + 1, offset):
                    end = start + sampling_points
                    window = filtered_data[:, start:end].T  # shape: (1920, 64)
                    
                    # Normalização aplicada por janela
                    window = normalize_sun(window)
                    
                    if self.transform:
                        window = self.transform(window)
                        
                    self.windows.append(window)
                    self.labels.append(label_map[subj])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Dados já estão filtrados e normalizados
        data, label = self.windows[idx], self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class EEGFixedTestDataset(Dataset):
    def __init__(self, folder_path, subjects, tasks=['R01', 'R02'], sampling_points=1920, label_map=None):
        self.windows = []
        self.labels = []
        self.fs = 160.0

        for subj in subjects:
            for task in tasks:
                file_name = f"S{subj:03d}{task}.csv"
                file_path = os.path.join(folder_path, f"S{subj:03d}", file_name)

                if not os.path.exists(file_path):
                    print(f"[AVISO] Arquivo não encontrado: {file_path}")
                    continue

                data = np.loadtxt(file_path, delimiter=',')  # shape: (64, total_amostras)
                # Aplicar filtro uma única vez por arquivo
                filtered_data = apply_fir_bandpass(data, fs=self.fs, lowcut=30.0, highcut=50.0, order=12)
                
                total_points = filtered_data.shape[1]

                step = (total_points - sampling_points) // 4
                for i in range(5):
                    start = i * step
                    end = start + sampling_points
                    window = filtered_data[:, start:end].T  # shape: (1920, 64)
                    
                    # Normalização aplicada por janela
                    window = normalize_sun(window)
                    
                    self.windows.append(window)
                    self.labels.append(label_map[subj])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Dados já estão filtrados e normalizados
        data, label = self.windows[idx], self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

batch_size = 100
learning_rate = 0.01
epochs = 40

# Todos os sujeitos (1-109)
all_subjects = list(range(1, 110))
num_classes = len(all_subjects)  # 109 classes
subject_names = [f"Subject {i}" for i in all_subjects]

# Mapeamento de sujeito para label (0-108)
subject_to_label = {subj: idx for idx, subj in enumerate(all_subjects)}

#Carrega todo o dataset
full_dataset = EEGWindowDataset(
    folder_path='/media/work/samuelbm/Dataset_CSV',
    subjects=all_subjects,
    tasks=['R05','R13'],                  
    sampling_points=1920,
    offset=35,
    label_map=subject_to_label
)

#Extrai todos os índices
indices = list(range(len(full_dataset)))
labels = [full_dataset.labels[i] for i in indices]

# Faz split estratificado (10% para validação)
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.1,
    stratify=labels,
    random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

test_dataset = EEGFixedTestDataset(
    folder_path='/media/work/samuelbm/Dataset_CSV',
    subjects=all_subjects,  # TODOS os sujeitos
    tasks=['R09'],
    sampling_points=1920,
    label_map=subject_to_label
)

# Nomes das classes para relatórios (109 sujeitos)
class_names = [f"Subject {i}" for i in all_subjects]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class EEG_BRNN_LSTM(nn.Module):
    def __init__(self,
                 input_channels=64,   # canais de entrada (ex.: 64)
                 time_steps=1920,     # passos temporais (ex.: 1920)
                 hidden_size=128,     # unidades LSTM por direção
                 num_layers=5,        # camadas empilhadas LSTM
                 bidirectional=True,  # BRNN -> bidirecional
                 num_classes=109,     # número de classes finais
                 ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # número de canais de saída temporal vindo da LSTM (por timestep)
        self.lstm_out_channels = hidden_size * (2 if bidirectional else 1)

        # 5 camadas de LSTM 
        self.lstm1 = nn.LSTM(
            input_size=input_channels,       # cada timestep tem dimensão = número de canais (64)
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.lstm2 = nn.LSTM(
            input_size=self.lstm_out_channels,       # cada timestep tem dimensão = número de canais (64)
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm3 = nn.LSTM(
            input_size=self.lstm_out_channels,       # cada timestep tem dimensão = número de canais (64)
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm4 = nn.LSTM(
            input_size=self.lstm_out_channels,       # cada timestep tem dime nsão = número de canais (64)
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm5 = nn.LSTM(
            input_size=self.lstm_out_channels,       # cada timestep tem dimensão = número de canais (64)
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # CNN para extração espacial/temporal (segue sua configuração anterior)
        self.cnn = nn.Sequential(
            # Conv1: in_channels = lstm_out_channels
            nn.Conv1d(self.lstm_out_channels, 96, kernel_size=11),   # -> [B, 96, L1]
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.MaxPool1d(kernel_size=4),                             # pool stride=4

            # Conv2
            nn.Conv1d(96, 128, kernel_size=9),                       # -> [B, 128, L2]
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),                             # pool stride=2

            # Conv3
            nn.Conv1d(128, 256, kernel_size=9),                      # -> [B, 256, L3]
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2),                             # pool stride=2 -> final length ~113
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(113)  # Força saída de tamanho 113

        # Classificador (mantive as dimensões que você usou)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # transforma [B, 256, 113] -> [B, 256*113]
            nn.Linear(256 * 113, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, time_steps, input_channels]
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out, _ = self.lstm5(out)

        # Para CNN: [B, C, T]
        x = out.permute(0, 2, 1)

        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEG_BRNN_LSTM(num_classes=num_classes, bidirectional=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[2, 37], gamma=0.1
)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=100):
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience, counter = 40, 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validação
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Época {epoch+1}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Carregar melhor modelo
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, test_loader, device, class_names, prefix="test"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Métricas principais
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(report)

    # ----- salvar métricas em arquivo (sem matriz de confusão) -----
    with open(f"{prefix}_metrics_insp.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # ----- gráfico de acurácia por classe -----
    num_classes = len(class_names)
    subject_acc = {}
    for i in range(num_classes):
        idx = np.where(np.array(all_labels) == i)[0]
        if len(idx) > 0:
            subject_acc[i] = (np.array(all_preds)[idx] == i).mean()
        else:
            subject_acc[i] = 0.0

    plt.figure(figsize=(18, 6))
    plt.bar(range(num_classes), [subject_acc[i] for i in range(num_classes)])
    plt.xlabel("Class ID")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Class")
    plt.xticks(range(num_classes), class_names, rotation=90, fontsize=6)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{prefix}_accuracy_insp.png")
    plt.close()

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, f1

model = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs
)

test_model(model, train_loader, device, subject_names, "train")

test_model(model, test_loader, device, subject_names)