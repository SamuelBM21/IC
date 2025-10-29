import torch                        #PyTorch
import torch.nn as nn               #Camadas da Rede neural
import torch.optim as optim         #Otimizadores
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from scipy.signal import firwin, filtfilt
import copy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

"""
Nome: parse_args
Função: Fazer parse dos argumentos da linha de comando para configuração do experimento
Entrada: Argumentos da linha de comando (sys.argv)
Saída: Objeto Namespace com todos os argumentos parseados
"""
def parse_args():
    parser = argparse.ArgumentParser(description='EEG Subject Identification with BRNN-LSTM')
    
    # Parâmetros de treinamento
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=40, help='Número de épocas (default: 40)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum do SGD (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (default: 0.0)')
    
    # Parâmetros do modelo
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size da LSTM (default: 128)')
    parser.add_argument('--num_layers', type=int, default=5, help='Número de camadas LSTM (default: 5)')
    parser.add_argument('--bidirectional', action='store_true', help='Usar LSTM bidirecional')
    parser.add_argument('--embedding_size', type=int, default=256, help='Tamanho do embedding final (default: 256)')
    
    # Parâmetros do dataset
    parser.add_argument('--data_path', type=str, default='/media/work/samuelbm/Dataset_CSV', 
                        help='Caminho para os dados')
    parser.add_argument('--train_tasks', type=str, nargs='+', default=['R05','R13'], 
                        help='Tasks de treino (pode passar múltiplas: --train_tasks R05 R06 R07)')
    parser.add_argument('--test_tasks', type=str, nargs='+', default=['R09'], 
                        help='Tasks de teste (pode passar múltiplas: --test_tasks R09 R10)')
    parser.add_argument('--sampling_points', type=int, default=1920, help='Pontos de amostragem (default: 1920)')
    parser.add_argument('--offset', type=int, default=35, help='Offset entre janelas (default: 35)')
    parser.add_argument('--val_split', type=float, default=0.1, help='Proporção de validação (default: 0.1)')
    
    # Parâmetros de filtro
    parser.add_argument('--lowcut', type=float, default=30.0, help='Frequência baixa do filtro (default: 30.0)')
    parser.add_argument('--highcut', type=float, default=50.0, help='Frequência alta do filtro (default: 50.0)')
    parser.add_argument('--filter_order', type=int, default=12, help='Ordem do filtro FIR (default: 12)')
    parser.add_argument('--fs', type=float, default=160.0, help='Frequência de amostragem (default: 160.0)')
    
    # Parâmetros de verificação
    parser.add_argument('--contrastive_margin', type=float, default=1.0, help='Margem da loss contrastiva (default: 1.0)')
    parser.add_argument('--num_test_pairs', type=int, default=2000, help='Número de pares para teste (default: 2000)')
    
    # Outros
    parser.add_argument('--patience', type=int, default=10, help='Patience para early stopping (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num_subjects', type=int, default=109, help='Número de sujeitos (default: 109)')
    parser.add_argument('--results_dir', type=str, default='eeg_verification_results', 
                        help='Diretório para salvar resultados')
    
    return parser.parse_args()

"""
Nome: set_seed
Função: Definir seed para reprodutibilidade dos experimentos
Entrada: seed (int) - valor da seed
Saída: --
"""
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
Nome: compute_eer
Função: Calcular o Equal Error Rate (EER) e o threshold ótimo
Entrada: y_true (array) - labels verdadeiros, distances (array) - distâncias calculadas
Saída: eer (float) - valor do EER, eer_threshold (float) - threshold ótimo
"""
def compute_eer(y_true, distances):
    fpr, tpr, thresholds = roc_curve(y_true, -distances)  # -distances pq menor = mais similar
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold

"""
Nome: apply_fir_bandpass
Função: Aplicar filtro FIR passa-banda no sinal EEG
Entrada: signal (array) - sinal de entrada, fs (float) - frequência de amostragem,
         lowcut (float) - frequência de corte inferior, highcut (float) - frequência de corte superior,
         order (int) - ordem do filtro
Saída: filtered_signal (array) - sinal filtrado
"""
def apply_fir_bandpass(signal, fs=160.0, lowcut=30.0, highcut=50.0, order=12):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # FIR filter usando firwin
    fir_coeff = firwin(order+1, [low, high], pass_zero=False)
    
    # Aplica filtro no último eixo
    filtered_signal = filtfilt(fir_coeff, 1.0, signal, axis=-1)
    
    return np.ascontiguousarray(filtered_signal)

"""
Nome: normalize_sun
Função: Aplicar normalização z-score por canal (método 'sun')
Entrada: signal (array) - sinal de entrada com shape (T, C)
Saída: signal normalizado (array) - sinal normalizado por canal
"""
def normalize_sun(signal):
    # Normalização por canal (cada coluna)
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    return (signal - mean) / (std + 1e-8)

"""
Nome: PairDataset
Função: Dataset que gera pares de amostras para treinamento contrastivo
Descrição: Cria pares positivos (mesma classe) e negativos (classes diferentes) para contrastive learning
"""
class PairDataset(Dataset):
    def __init__(self, base_dataset):
        """
        Entrada: base_dataset - dataset base com amostras e labels
        """
        self.base_dataset = base_dataset

        # Extrair labels alinhados ao índice usado por base_dataset[idx]
        if hasattr(base_dataset, 'labels'):
            # dataset "plano"
            labels = np.array(base_dataset.labels, dtype=np.int32)
        elif hasattr(base_dataset, 'dataset') and hasattr(base_dataset, 'indices'):
            # Subset: alinhar os labels aos indices do subset
            labels = np.array([base_dataset.dataset.labels[i] for i in base_dataset.indices], dtype=np.int32)
        else:
            raise AttributeError("base_dataset deve ter 'labels' ou ser Subset com 'dataset' e 'indices'")

        self.labels = labels
        self.length = len(self.labels)

        self.indices_by_label = {}
        for i, lab in enumerate(self.labels):
            lab = int(lab)
            if lab in self.indices_by_label:
                self.indices_by_label[lab].append(i)
            else:
                self.indices_by_label[lab] = [i]

        self.unique_labels = np.array(list(self.indices_by_label.keys()), dtype=np.int32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Saída: x1, x2 (tensors) - par de amostras, label (tensor) - 1.0 para par positivo, 0.0 para negativo
        """
        # pegar primeiro item (base_dataset já mapeia Subset corretamente)
        x1, y1 = self.base_dataset[idx]
        y1 = int(y1)

        if np.random.rand() < 0.5:
            # Par positivo (mesma classe)
            candidates = self.indices_by_label[y1]
            if len(candidates) > 1:
                while True:
                    idx2 = np.random.choice(candidates)
                    if idx2 != idx:
                        break
            else:
                idx2 = idx
            label = 1.0
        else:
            # Par negativo (classes diferentes)
            if len(self.unique_labels) > 1:
                while True:
                    other_label = int(np.random.choice(self.unique_labels))
                    if other_label != y1:
                        break
                idx2 = np.random.choice(self.indices_by_label[other_label])
            else:
                idx2 = idx
            label = 0.0

        x2, _ = self.base_dataset[idx2]
        return x1, x2, torch.tensor(label, dtype=torch.float32)

"""
Nome: ContrastiveLoss
Função: Implementar a função de perda contrastiva para aprendizado de embeddings
Descrição: Loss que aproxima pares similares e afasta pares diferentes
"""
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Entrada: margin (float) - margem para pares negativos
        """
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        """
        Entrada: z1, z2 (tensors) - embeddings dos pares, label (tensor) - 1 para similar, 0 para diferente
        Saída: loss (tensor) - valor da perda contrastiva
        """
        d = torch.nn.functional.pairwise_distance(z1, z2)
        loss = torch.mean(label * d.pow(2) +
                          (1 - label) * torch.clamp(self.margin - d, min=0).pow(2))
        return loss

"""
Nome: EEGWindowDataset
Função: Dataset para carregar janelas de sinais EEG com filtragem e normalização
Descrição: Carrega arquivos CSV, aplica filtro FIR e normalização, cria janelas deslizantes
"""
class EEGWindowDataset(Dataset):
    def __init__(self, folder_path, subjects, tasks, sampling_points=1920, offset=35, 
                 label_map=None, transform=None, fs=160.0, lowcut=30.0, highcut=50.0, filter_order=12):
        """
        Entrada: folder_path (str) - caminho dos dados, subjects (list) - lista de sujeitos,
                 tasks (list) - lista de tasks, sampling_points (int) - tamanho da janela,
                 offset (int) - deslocamento entre janelas, label_map (dict) - mapeamento sujeito->label,
                 transform (callable) - transformações adicionais, fs (float) - frequência de amostragem,
                 lowcut/highcut (float) - frequências de corte do filtro, filter_order (int) - ordem do filtro
        """
        self.windows = []
        self.labels = []
        self.transform = transform
        self.fs = fs

        for subj in subjects:
            for task in tasks:
                file_name = f"S{subj:03d}{task}.csv"
                file_path = os.path.join(folder_path, f"S{subj:03d}", file_name)

                if not os.path.exists(file_path):
                    print(f"[AVISO] Arquivo não encontrado: {file_path}")
                    continue

                data = np.loadtxt(file_path, delimiter=',')  # shape: (64, total_amostras)
                # Aplicar filtro uma única vez por arquivo
                filtered_data = apply_fir_bandpass(data, fs=self.fs, lowcut=lowcut, 
                                                   highcut=highcut, order=filter_order)
                
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
        """
        Saída: data (tensor) - janela de EEG, label (tensor) - label do sujeito
        """
        data, label = self.windows[idx], self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

"""
Nome: EEGFixedTestDataset
Função: Dataset para teste com janelas fixas de sinais EEG
Descrição: Similar ao EEGWindowDataset mas com 5 janelas fixas por arquivo para teste consistente
"""
class EEGFixedTestDataset(Dataset):
    def __init__(self, folder_path, subjects, tasks=['R01', 'R02'], sampling_points=1920, 
                 label_map=None, fs=160.0, lowcut=30.0, highcut=50.0, filter_order=12):
        """
        Entrada: folder_path (str) - caminho dos dados, subjects (list) - lista de sujeitos,
                 tasks (list) - lista de tasks de teste, sampling_points (int) - tamanho da janela,
                 label_map (dict) - mapeamento sujeito->label, fs (float) - frequência de amostragem,
                 lowcut/highcut (float) - frequências de corte do filtro, filter_order (int) - ordem do filtro
        """
        self.windows = []
        self.labels = []
        self.fs = fs

        for subj in subjects:
            for task in tasks:
                file_name = f"S{subj:03d}{task}.csv"
                file_path = os.path.join(folder_path, f"S{subj:03d}", file_name)

                if not os.path.exists(file_path):
                    print(f"[AVISO] Arquivo não encontrado: {file_path}")
                    continue

                data = np.loadtxt(file_path, delimiter=',')  # shape: (64, total_amostras)
                # Aplicar filtro uma única vez por arquivo
                filtered_data = apply_fir_bandpass(data, fs=self.fs, lowcut=lowcut, 
                                                   highcut=highcut, order=filter_order)
                
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
        """
        Saída: data (tensor) - janela de EEG, label (tensor) - label do sujeito
        """
        data, label = self.windows[idx], self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

"""
Nome: EEG_BRNN_LSTM
Função: Modelo de rede neural com LSTM (bidirecional ou unidirecional) + CNN para extração de embeddings
Descrição: 5 camadas LSTM seguidas de CNN 1D e camadas fully connected para gerar embeddings normalizados
"""
class EEG_BRNN_LSTM(nn.Module):
    def __init__(self,
                 input_channels=64,   # canais de entrada (ex.: 64)
                 time_steps=1920,     # passos temporais (ex.: 1920)
                 hidden_size=128,     # unidades LSTM por direção
                 num_layers=5,        # camadas empilhadas LSTM
                 bidirectional=True,  # BRNN -> bidirecional
                 num_classes=109,     # número de classes finais
                 lstm_dropout=0.0,    # dropout entre camadas LSTM (se >0 e num_layers>1)
                 embedding_size=256   # tamanho do embedding final
                 ):
        """
        Entrada: input_channels (int) - número de canais EEG, time_steps (int) - tamanho temporal,
                 hidden_size (int) - tamanho hidden LSTM, num_layers (int) - número de camadas,
                 bidirectional (bool) - usar LSTM bidirecional, num_classes (int) - número de sujeitos,
                 lstm_dropout (float) - dropout LSTM, embedding_size (int) - tamanho do embedding
        """
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        # número de canais de saída temporal vindo da LSTM (por timestep)
        self.lstm_out_channels = hidden_size * (2 if bidirectional else 1)

        # 5 camadas de LSTM 
        self.lstm1 = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.lstm2 = nn.LSTM(
            input_size=self.lstm_out_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm3 = nn.LSTM(
            input_size=self.lstm_out_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm4 = nn.LSTM(
            input_size=self.lstm_out_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm5 = nn.LSTM(
            input_size=self.lstm_out_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.cnn = nn.Sequential(
            # Conv1: in_channels = lstm_out_channels
            nn.Conv1d(self.lstm_out_channels, 96, kernel_size=11),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.MaxPool1d(kernel_size=4),

            # Conv2
            nn.Conv1d(96, 128, kernel_size=9),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),

            # Conv3
            nn.Conv1d(128, 256, kernel_size=9),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(113)

        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 113, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.embedding_size),
        )

    def forward(self, x):
        """
        Entrada: x (tensor) - batch de janelas EEG [B, time_steps, input_channels]
        Saída: embedding normalizado (tensor) - [B, embedding_size]
        """
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
        x = self.features(x)
        return nn.functional.normalize(x, p=2, dim=1)

"""
Nome: train_verification
Função: Treinar modelo de verificação usando contrastive learning
Entrada: model - modelo neural, train_loader - loader de treino, val_loader - loader de validação,
         optimizer - otimizador, device - dispositivo (CPU/GPU), epochs (int) - número de épocas
Saída: model treinado, training_history (dict) - histórico de treinamento
"""
def train_verification(model, train_loader, val_loader, optimizer, scheduler, device, epochs=40):
    criterion = ContrastiveLoss(margin=1.0)
    best_eer = 1.0
    best_wts = copy.deepcopy(model.state_dict())
    
    # Listas para armazenar histórico de treinamento
    train_losses = []
    val_eers = []
    epoch_logs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = criterion(z1, z2, labels)  
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Atualizar o scheduler após cada época
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        if current_lr != new_lr:
            print(f"LR atualizado: {current_lr:.6f} -> {new_lr:.6f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Avaliação
        model.eval()
        all_labels, all_distances = [], []
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2 = x1.to(device), x2.to(device)
                z1, z2 = model(x1), model(x2)
                d = torch.nn.functional.pairwise_distance(z1, z2).cpu().numpy()
                all_labels.extend(labels.numpy())
                all_distances.extend(d)

        eer, _ = compute_eer(np.array(all_labels), np.array(all_distances))
        val_eers.append(eer)
        
        # Log da época
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_eer': eer,
            'is_best': eer < best_eer
        }
        epoch_logs.append(epoch_log)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss {avg_train_loss:.4f} | EER {eer:.4f}")

        if eer < best_eer:
            best_eer = eer
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    print(f"Best EER: {best_eer:.4f}")
    
    # Retorna também o histórico de treinamento
    training_history = {
        'train_losses': train_losses,
        'val_eers': val_eers,
        'epoch_logs': epoch_logs,
        'best_eer': best_eer
    }
    
    return model, training_history

"""
Nome: test_verification
Função: Testar modelo de verificação comparando todos contra todos e calculando métricas
Entrada: model - modelo treinado, test_dataset - dataset de teste, device - dispositivo
Saída: test_results (dict) - dicionário com todas as métricas de teste
"""
def test_verification(model, test_dataset, device, save_dir="results", save_matrix=True):
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    print("Extraindo embeddings do conjunto de teste...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, label = test_dataset[i]
            x = x.unsqueeze(0).to(device)
            z = model(x)
            all_embeddings.append(z.cpu().numpy().flatten())
            all_labels.append(label)

    embeddings = np.stack(all_embeddings)
    labels = np.array(all_labels)

    print("Calculando matriz de similaridade...")
    sim_matrix = cosine_similarity(embeddings)
    if save_matrix:
        matrix_path = os.path.join(save_dir, "similarity_matrix.csv")
        np.savetxt(matrix_path, sim_matrix, delimiter=",")
        print(f"Matriz de similaridade salva em {matrix_path}")

    genuine_scores = []
    impostor_scores = []
    all_distances = []  # Para armazenar todas as distâncias
    pair_labels = []    # Para armazenar os labels dos pares

    print("Calculando scores de genuínos e impostores...")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = sim_matrix[i, j]
            # Converter similaridade em distância: distância = 1 - similaridade
            distance = 1.0 - score
            
            all_distances.append(distance)
            
            if labels[i] == labels[j]:
                genuine_scores.append(score)
                pair_labels.append(1)  # Par genuíno
            else:
                impostor_scores.append(score)
                pair_labels.append(0)  # Par impostor

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    all_distances = np.array(all_distances)
    pair_labels = np.array(pair_labels)
    
    # Calcular EER usando scores (não distâncias)
    # Para similaridade, valores ALTOS = mesma pessoa, então não invertemos
    print("Calculando EER...")
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    roc_auc = auc(fpr, tpr)
    
    # Calcular acurácia no threshold ótimo
    predictions = (y_scores >= eer_threshold).astype(int)
    accuracy = np.mean(predictions == y_true)

    # Salvar gráficos
    print("Gerando gráficos...")
    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.plot(eer, 1-eer, 'ro', markersize=10, label=f'EER = {eer:.3f}')
    plt.xlabel("Taxa de Falsa Aceitação (FAR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TAR)")
    plt.title("Curva ROC - Verificação EEG")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve salva em {roc_path}")

    # Histogramas de scores
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.7, label=f'Genuíno (n={len(genuine_scores)})', 
             color='green', density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.7, label=f'Impostor (n={len(impostor_scores)})', 
             color='red', density=True)
    plt.axvline(eer_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'Threshold = {eer_threshold:.3f}')
    plt.xlabel("Similaridade (Cosseno)")
    plt.ylabel("Densidade")
    plt.title("Distribuição de Similaridades")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(save_dir, "score_histograms.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histograma salvo em {hist_path}")

    # Criar dicionário de resultados com TODAS as informações necessárias
    results = {
        'eer': float(eer),
        'threshold': float(eer_threshold),
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'num_genuine_pairs': int(len(genuine_scores)),
        'num_impostor_pairs': int(len(impostor_scores)),
        'mean_genuine_sim': float(np.mean(genuine_scores)),
        'std_genuine_sim': float(np.std(genuine_scores)),
        'mean_impostor_sim': float(np.mean(impostor_scores)),
        'std_impostor_sim': float(np.std(impostor_scores)),
        # Adicionar como distâncias também para compatibilidade
        'mean_genuine_distance': float(1.0 - np.mean(genuine_scores)),
        'std_genuine_distance': float(np.std(1.0 - genuine_scores)),
        'mean_impostor_distance': float(1.0 - np.mean(impostor_scores)),
        'std_impostor_distance': float(np.std(1.0 - impostor_scores)),
        # Para os gráficos
        'all_distances': all_distances.tolist(),
        'all_labels': pair_labels.tolist(),
        # Dados da ROC (convertidos para lista para serialização JSON)
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_thresholds': thresholds.tolist(),
    }
    
    print("\n" + "="*60)
    print("RESULTADOS DO TESTE (All-vs-All)")
    print("="*60)
    print(f"EER: {eer*100:.2f}%")
    print(f"Threshold ótimo: {eer_threshold:.4f}")
    print(f"Acurácia: {accuracy*100:.2f}%")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Similaridade média (genuíno): {np.mean(genuine_scores):.3f} ± {np.std(genuine_scores):.3f}")
    print(f"Similaridade média (impostor): {np.mean(impostor_scores):.3f} ± {np.std(impostor_scores):.3f}")
    print(f"Pares genuínos: {len(genuine_scores)}")
    print(f"Pares impostores: {len(impostor_scores)}")
    print("="*60)

    return results

"""
Nome: save_plots
Função: Salvar gráficos de treinamento e teste (loss, EER, ROC, histograma de distâncias)
Entrada: training_history (dict) - histórico de treinamento, test_results (dict) - resultados de teste,
         experiment_name (str) - nome do experimento, results_dir (str) - diretório para salvar
Saída: --
"""
def save_plots(training_history, test_results, experiment_name, results_dir):
    """
    Salva gráficos de treinamento
    """
    print("Salvando gráficos de treinamento...")
    
    # Gráfico de perda de treinamento e EER de validação
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(training_history['train_losses']) + 1)
    
    ax1.plot(epochs, training_history['train_losses'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Perda Contrastiva')
    ax1.set_title('Perda de Treinamento')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, training_history['val_eers'], 'r-', linewidth=2, label='Validation EER')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Equal Error Rate (EER)')
    ax2.set_title('EER de Validação')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    training_path = os.path.join(results_dir, f'{experiment_name}_training_curves.png')
    plt.savefig(training_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curvas de treinamento salvas em {training_path}")
    
    # Gráfico de distâncias (se houver dados disponíveis)
    if 'all_distances' in test_results and 'all_labels' in test_results:
        plt.figure(figsize=(10, 6))
        
        distances = np.array(test_results['all_distances'])
        labels = np.array(test_results['all_labels'])
        
        positive_distances = distances[labels == 1]
        negative_distances = distances[labels == 0]
        
        if len(positive_distances) > 0 and len(negative_distances) > 0:
            plt.hist(positive_distances, bins=50, alpha=0.7, 
                    label=f'Mesmo Sujeito (n={len(positive_distances)})', 
                    color='green', density=True)
            plt.hist(negative_distances, bins=50, alpha=0.7, 
                    label=f'Sujeitos Diferentes (n={len(negative_distances)})', 
                    color='red', density=True)
            
            # Threshold baseado em distância (1 - threshold de similaridade)
            distance_threshold = 1.0 - test_results['threshold']
            plt.axvline(distance_threshold, color='black', linestyle='--', linewidth=2, 
                       label=f'Threshold ({distance_threshold:.4f})')
            
            plt.xlabel('Distância Euclidiana')
            plt.ylabel('Densidade')
            plt.title('Distribuição de Distâncias - Verificação EEG')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            dist_path = os.path.join(results_dir, f'{experiment_name}_distance_histogram.png')
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Histograma de distâncias salvo em {dist_path}")

"""
Nome: save_results
Função: Salvar todos os resultados do experimento (configurações, histórico, métricas, modelo, gráficos)
Entrada: training_history (dict) - histórico de treinamento, test_results (dict) - resultados de teste,
         model - modelo treinado, experiment_name (str) - nome do experimento,
         timestamp (str) - timestamp da execução, results_dir (str) - diretório para salvar,
         config_info (dict) - informações de configuração
Saída: --
"""
def save_results(training_history, test_results, model, experiment_name, 
                 timestamp, results_dir, config_info):
    """
    Salva todos os resultados do experimento
    """
    print("\n" + "="*60)
    print("SALVANDO RESULTADOS")
    print("="*60)
    
    try:
        # 1. Configuração do experimento
        config = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'model_architecture': {
                'input_channels': 64,
                'time_steps': config_info['sampling_points'],
                'hidden_size': config_info['hidden_size'],
                'num_layers': config_info['num_layers'],
                'bidirectional': config_info['bidirectional'],
                'embedding_size': config_info['embedding_size']
            },
            'training_config': {
                'batch_size': config_info['batch_size'],
                'learning_rate': config_info['learning_rate'],
                'momentum': config_info['momentum'],
                'epochs': config_info['epochs'],
                'train_tasks': config_info['train_tasks'],
                'test_tasks': config_info['test_tasks'],
                'num_subjects': config_info['num_subjects'],
                'sampling_points': config_info['sampling_points'],
                'offset': config_info['offset'],
                'val_split': config_info['val_split']
            },
            'filter_config': {
                'fs': config_info['fs'],
                'lowcut': config_info['lowcut'],
                'highcut': config_info['highcut'],
                'filter_order': config_info['filter_order']
            },
            'dataset_info': {
                'total_subjects': config_info['total_subjects'],
                'train_size': config_info['train_size'],
                'val_size': config_info['val_size'],
                'test_size': config_info['test_size']
            }
        }
        
        config_path = os.path.join(results_dir, f'{experiment_name}_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"✓ Configuração salva: {config_path}")
        
    except Exception as e:
        print(f"✗ Erro ao salvar configuração: {e}")
    
    try:
        # 2. Histórico de treinamento
        history_path = os.path.join(results_dir, f'{experiment_name}_training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=4, ensure_ascii=False)
        print(f"✓ Histórico de treinamento salvo: {history_path}")
        
    except Exception as e:
        print(f"✗ Erro ao salvar histórico: {e}")
    
    try:
        # 3. Resultados de teste (remover dados muito grandes para JSON)
        test_results_compact = test_results.copy()
        # Manter apenas resumos estatísticos, não listas enormes
        if 'fpr' in test_results_compact:
            test_results_compact['fpr'] = 'Dados salvos em arquivo separado'
        if 'tpr' in test_results_compact:
            test_results_compact['tpr'] = 'Dados salvos em arquivo separado'
        if 'roc_thresholds' in test_results_compact:
            test_results_compact['roc_thresholds'] = 'Dados salvos em arquivo separado'
        if 'all_distances' in test_results_compact:
            test_results_compact['all_distances'] = 'Dados salvos em arquivo separado'
        if 'all_labels' in test_results_compact:
            test_results_compact['all_labels'] = 'Dados salvos em arquivo separado'
            
        results_path = os.path.join(results_dir, f'{experiment_name}_test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results_compact, f, indent=4, ensure_ascii=False)
        print(f"✓ Resultados de teste salvos: {results_path}")
        
    except Exception as e:
        print(f"✗ Erro ao salvar resultados de teste: {e}")
    
    try:
        # 4. Modelo treinado
        model_path = os.path.join(results_dir, f'{experiment_name}_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config['model_architecture'],
            'best_val_eer': training_history['best_eer'],
            'test_eer': test_results['eer']
        }, model_path)
        print(f"✓ Modelo salvo: {model_path}")
        
    except Exception as e:
        print(f"✗ Erro ao salvar modelo: {e}")
    
    try:
        # 5. Relatório resumido
        summary_report = {
            'Experimento': experiment_name,
            'Data': timestamp,
            'Treinamento': {
                'Melhor EER de Validação': f"{training_history['best_eer']:.4f}",
                'Perda Final de Treinamento': f"{training_history['train_losses'][-1]:.4f}",
                'Total de Épocas': len(training_history['train_losses'])
            },
            'Resultados de Teste': {
                'EER de Teste': f"{test_results['eer']:.4f}",
                'Threshold Ótimo': f"{test_results['threshold']:.4f}",
                'Acurácia de Teste': f"{test_results['accuracy']:.4f}",
                'AUC': f"{test_results.get('roc_auc', 0):.4f}",
                'Similaridade Genuíno (média±std)': f"{test_results['mean_genuine_sim']:.4f}±{test_results['std_genuine_sim']:.4f}",
                'Similaridade Impostor (média±std)': f"{test_results['mean_impostor_sim']:.4f}±{test_results['std_impostor_sim']:.4f}",
                'Número de Pares Genuínos': test_results['num_genuine_pairs'],
                'Número de Pares Impostores': test_results['num_impostor_pairs']
            },
            'Dataset': {
                'Total de Sujeitos': config['dataset_info']['total_subjects'],
                'Amostras de Treino': config['dataset_info']['train_size'],
                'Amostras de Validação': config['dataset_info']['val_size'],
                'Amostras de Teste': config['dataset_info']['test_size']
            }
        }
        
        summary_path = os.path.join(results_dir, f'{experiment_name}_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        print(f"✓ Resumo salvo: {summary_path}")
        
    except Exception as e:
        print(f"✗ Erro ao salvar resumo: {e}")
    
    try:
        # 6. Gráficos
        save_plots(training_history, test_results, experiment_name, results_dir)
        print(f"✓ Gráficos salvos")
        
    except Exception as e:
        print(f"✗ Erro ao salvar gráficos: {e}")
    
    print("="*60)
    print(f"Resultados salvos em: {os.path.abspath(results_dir)}/")
    print("="*60)
    
"""
Nome: main
Função: Função principal que orquestra todo o pipeline de treinamento e teste
Entrada: Argumentos da linha de comando (via parse_args)
Saída: --
"""
def main():
    args = parse_args()
    set_seed(args.seed)

    # Criar diretório para salvar os resultados
    os.makedirs(args.results_dir, exist_ok=True)

    # Timestamp para identificar a execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"eeg_verification_{timestamp}"
        
    print("=" * 80)
    print("CONFIGURAÇÕES DO EXPERIMENTO")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 80)
    
    # Todos os sujeitos (1-109)
    all_subjects = list(range(1, args.num_subjects + 1))
    num_classes = len(all_subjects)
    subject_names = [f"Subject {i}" for i in all_subjects]

    # Mapeamento de sujeito para label (0-108)
    subject_to_label = {subj: idx for idx, subj in enumerate(all_subjects)}

    # Carrega todo o dataset
    full_dataset = EEGWindowDataset(
        folder_path=args.data_path,
        subjects=all_subjects,
        tasks=args.train_tasks,                  
        sampling_points=args.sampling_points,
        offset=args.offset,
        label_map=subject_to_label,
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order
    )

    # Extrai todos os índices
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.labels[i] for i in indices]

    # Faz split estratificado
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        stratify=labels,
        random_state=args.seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    test_dataset = EEGFixedTestDataset(
        folder_path=args.data_path,
        subjects=all_subjects,
        tasks=args.test_tasks,
        sampling_points=args.sampling_points,
        label_map=subject_to_label,
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order
    )

    # Criar PairDatasets para treinamento contrastivo
    train_pair_dataset = PairDataset(train_dataset)
    val_pair_dataset = PairDataset(val_dataset)

    train_pair_loader = DataLoader(train_pair_dataset, batch_size=args.batch_size, shuffle=True)
    val_pair_loader = DataLoader(val_pair_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nTamanho do conjunto de treino: {len(train_dataset)}")
    print(f"Tamanho do conjunto de validação: {len(val_dataset)}")
    print(f"Tamanho do conjunto de teste: {len(test_dataset)}")

    # Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}")
    
    model = EEG_BRNN_LSTM(
        input_channels=64,
        time_steps=args.sampling_points,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        num_classes=num_classes,
        embedding_size=args.embedding_size
    ).to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.learning_rate, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 37], gamma=0.1
    )
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")

    # Treinar
    print("\n" + "=" * 80)
    print("INICIANDO TREINAMENTO")
    print("=" * 80)
    
    model, training_history = train_verification(
        model,
        train_pair_loader,
        val_pair_loader,
        optimizer,
        scheduler,
        device,
        epochs=args.epochs
    )

    print("\n" + "=" * 80)
    print("INICIANDO TESTE")
    print("=" * 80)
    
    test_results = test_verification(model, test_dataset, device, args.results_dir,True)
    
    # Criar dicionário com informações de configuração
    config_info = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'epochs': args.epochs,
        'train_tasks': args.train_tasks,
        'test_tasks': args.test_tasks,
        'num_subjects': args.num_subjects,
        'total_subjects': len(all_subjects),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'sampling_points': args.sampling_points,
        'offset': args.offset,
        'val_split': args.val_split,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'bidirectional': args.bidirectional,
        'embedding_size': args.embedding_size,
        'fs': args.fs,
        'lowcut': args.lowcut,
        'highcut': args.highcut,
        'filter_order': args.filter_order
    }
    
    # Salvar resultados
    save_results(training_history, test_results, model, experiment_name,
                 timestamp, args.results_dir, config_info)
    
    print("\n" + "=" * 80)
    print("EXPERIMENTO FINALIZADO COM SUCESSO!")
    print("=" * 80)

if __name__ == "__main__":
    main()