import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchxlstm import xLSTM
import copy
from sklearn.model_selection import train_test_split
from scipy.signal import firwin, filtfilt
import argparse
from tqdm import tqdm

def parse_args():
    """Parse argumentos da linha de comando"""
    parser = argparse.ArgumentParser(description='EEG Subject Identification with xLSTM')
    
    # Parâmetros de treinamento
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas (default: 100)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (default: 0.0001)')
    
    # Parâmetros do modelo
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size da xLSTM (default: 128)')
    parser.add_argument('--num_layers', type=int, default=3, help='Número de camadas xLSTM (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
    
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
    
    # Outros
    parser.add_argument('--patience', type=int, default=10, help='Patience para early stopping (default: 10)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num_subjects', type=int, default=109, help='Número de sujeitos (default: 109)')
    
    return parser.parse_args()

def set_seed(seed):
    """Define seed para reprodutibilidade"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_fir_bandpass(signal, fs=160.0, lowcut=30.0, highcut=50.0, order=12):
    """Aplica filtro FIR passa-banda"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    fir_coeff = firwin(order+1, [low, high], pass_zero=False)
    filtered_signal = filtfilt(fir_coeff, 1.0, signal, axis=-1)
    
    return np.ascontiguousarray(filtered_signal)

def normalize_sun(signal):
    """Normalização 'sun' por canal"""
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    return (signal - mean) / (std + 1e-8)

class EEGWindowDataset(Dataset):
    def __init__(self, folder_path, subjects, tasks, sampling_points=1920, offset=35, 
                 label_map=None, lowcut=30.0, highcut=50.0, filter_order=12):
        self.windows = []
        self.labels = []
        self.fs = 160.0
        
        print(f"Carregando dataset com {len(subjects)} sujeitos e tasks {tasks}...")
        
        for subj in tqdm(subjects, desc="Processando sujeitos"):
            for task in tasks:
                file_name = f"S{subj:03d}{task}.csv"
                file_path = os.path.join(folder_path, f"S{subj:03d}", file_name)

                if not os.path.exists(file_path):
                    print(f"[AVISO] Arquivo não encontrado: {file_path}")
                    continue

                data = np.loadtxt(file_path, delimiter=',')
                filtered_data = apply_fir_bandpass(data, fs=self.fs, lowcut=lowcut, 
                                                   highcut=highcut, order=filter_order)
                
                total_points = filtered_data.shape[1]

                for start in range(0, total_points - sampling_points + 1, offset):
                    end = start + sampling_points
                    window = filtered_data[:, start:end].T  # (1920, 64)
                    window = normalize_sun(window)
                    
                    self.windows.append(window)
                    self.labels.append(label_map[subj])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data, label = self.windows[idx], self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class EEGFixedTestDataset(Dataset):
    def __init__(self, folder_path, subjects, tasks, sampling_points=1920, label_map=None,
                 lowcut=30.0, highcut=50.0, filter_order=12):
        self.windows = []
        self.labels = []
        self.fs = 160.0
        
        print(f"Carregando dataset de teste com {len(subjects)} sujeitos e tasks {tasks}...")

        for subj in tqdm(subjects, desc="Processando sujeitos (teste)"):
            for task in tasks:
                file_name = f"S{subj:03d}{task}.csv"
                file_path = os.path.join(folder_path, f"S{subj:03d}", file_name)

                if not os.path.exists(file_path):
                    print(f"[AVISO] Arquivo não encontrado: {file_path}")
                    continue

                data = np.loadtxt(file_path, delimiter=',')
                filtered_data = apply_fir_bandpass(data, fs=self.fs, lowcut=lowcut, 
                                                   highcut=highcut, order=filter_order)
                
                total_points = filtered_data.shape[1]
                step = (total_points - sampling_points) // 4

                for i in range(5):
                    start = i * step
                    end = start + sampling_points
                    window = filtered_data[:, start:end].T
                    window = normalize_sun(window)
                    
                    self.windows.append(window)
                    self.labels.append(label_map[subj])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        data, label = self.windows[idx], self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class EEG_CNN_xLSTM(nn.Module):
    """
    CNN + xLSTM simplificado para sequências longas (stride adicionado).
    Mantém hiperparâmetros e lógica original, mas reduz o tamanho temporal
    antes da xLSTM para economia de memória e tempo.
    """
    def __init__(self, input_size=64, hidden_size=96, num_classes=109,
                 xlstm_layers=('s',)):
        super().__init__()

        # ----- 1. Bloco convolucional inicial -----
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.hidden_size = hidden_size

        # ----- 2. Bloco convolucional com stride -----
        # stride=4 reduz o tamanho temporal de 1920 -> 480
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, self.hidden_size, kernel_size=3, padding=1, stride=4),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Ajustar residual para combinar dimensões temporais reduzidas
        self.residual_conv = nn.Sequential(
            nn.Conv1d(128, self.hidden_size, kernel_size=1, stride=4),
            nn.BatchNorm1d(self.hidden_size)
        )

        head_size = hidden_size
        num_heads = 1
        self.xlstm_layers = list(xlstm_layers)

        self.xlstm = xLSTM(
            input_size=self.hidden_size,
            head_size=head_size,
            num_heads=num_heads,
            layers=self.xlstm_layers,
            batch_first=True,
        )

        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, C, T]
        x = x.permute(0, 2, 1)  # [B, C, T]

        out = self.conv_block1(x)  # [B, 128, T]
        residual = self.residual_conv(out)  # [B, hidden_size, T/stride]

        out = self.conv_block2(out) + residual  # [B, hidden_size, T/stride]
        out = out.permute(0, 2, 1)  # [B, T/stride, hidden_size]

        out, _ = self.xlstm(out)  # [B, T/stride, hidden_size]
        out = out.permute(0, 2, 1)  # [B, hidden_size, T/stride]

        out = self.temporal_pooling(out).squeeze(-1)  # [B, hidden_size]
        return self.classifier(out)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience):
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0
    
    train_losses = []
    val_losses = []
    val_f1s = []

    for epoch in range(epochs):
        # Treino
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validação
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        # Scheduler step
        scheduler.step(val_loss)

        print(f"Época {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            # Salvar melhor modelo
            torch.save(model.state_dict(), 'best_model_xlstm.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping na época {epoch+1}")
                break

    # Carregar melhor modelo
    model.load_state_dict(best_model_wts)
    
    # Plotar curvas de aprendizado
    plot_training_curves(train_losses, val_losses, val_f1s)
    
    return model

def plot_training_curves(train_losses, val_losses, val_f1s):
    """Plota curvas de aprendizado"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.set_title('Curvas de Loss')
    ax1.legend()
    ax1.grid(True)
    
    # F1
    ax2.plot(val_f1s, label='Val F1', color='green')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score na Validação')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_xlstm.png', dpi=300)
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

def test_model(model, test_loader, device, class_names, prefix="test"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testando"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Métricas
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(report)

    # Salvar métricas
    with open(f"{prefix}_metrics_xlstm.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # Gráfico de acurácia por classe
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
    plt.savefig(f"{prefix}_accuracy_xlstm.png", dpi=300)
    plt.close()
    
    return accuracy

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 80)
    print("CONFIGURAÇÕES DO EXPERIMENTO")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 80)
    
    # Configurar sujeitos
    all_subjects = list(range(1, args.num_subjects + 1))
    num_classes = len(all_subjects)
    subject_names = [f"Subject {i}" for i in all_subjects]
    subject_to_label = {subj: idx for idx, subj in enumerate(all_subjects)}

    # Carregar dataset
    print(f"\nCarregando dataset de TREINO com tasks: {args.train_tasks}")
    full_dataset = EEGWindowDataset(
        folder_path=args.data_path,
        subjects=all_subjects,
        tasks=args.train_tasks, 
        sampling_points=args.sampling_points,
        offset=args.offset,
        label_map=subject_to_label,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order
    )

    # Split estratificado
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.labels[i] for i in indices]
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        stratify=labels,
        random_state=args.seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Dataset de teste
    print(f"\nCarregando dataset de TESTE com tasks: {args.test_tasks}")
    test_dataset = EEGFixedTestDataset(
        folder_path=args.data_path,
        subjects=all_subjects,
        tasks=args.test_tasks,  
        sampling_points=args.sampling_points,
        label_map=subject_to_label,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nTamanho do conjunto de treino: {len(train_dataset)}")
    print(f"Tamanho do conjunto de validação: {len(val_dataset)}")
    print(f"Tamanho do conjunto de teste: {len(test_dataset)}")

    # Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}")
    
    model = EEG_CNN_xLSTM(
        input_size=64,
        hidden_size=args.hidden_size,
        num_classes=num_classes,
    ).to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")

    # Otimizador e loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Treinar
    print("\n" + "=" * 80)
    print("INICIANDO TREINAMENTO")
    print("=" * 80)
    
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        args.epochs,
        args.patience
    )

    # Testar
    print("\n" + "=" * 80)
    print("DESEMPENHO NO CONJUNTO DE TREINO")
    print("=" * 80)
    train_acc = test_model(model, train_loader, device, subject_names, prefix="train")

    print("\n" + "=" * 80)
    print("DESEMPENHO NO CONJUNTO DE TESTE")
    print("=" * 80)
    test_acc = test_model(model, test_loader, device, subject_names, prefix="test")
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS FINAIS")
    print(f"{'='*80}")
    print(f"Acurácia de Treino: {train_acc:.4f}")
    print(f"Acurácia de Teste: {test_acc:.4f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()