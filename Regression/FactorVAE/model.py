import torch

from sklearn.metrics import r2_score
import numpy as np
from tqdm.auto import tqdm

def train(factor_model, dataloader, optimizer, scheduler, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.train()
    total_loss = 0
    with tqdm(total=len(dataloader), desc="Training") as pbar:
        for char_with_label, _ in dataloader:
            char = char_with_label[:,:,:-1]
            returns = char_with_label[:,:,-1]

            inputs = char.to(device)
            labels = returns[:,-1].reshape(-1,1).to(device)
            inputs = inputs.float()
            labels = labels.float()
            
            optimizer.zero_grad()
            loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factor_model(inputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            pbar.set_postfix({'batch_loss': loss.item()})
            pbar.update(1)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(factor_model, dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()

    total_loss = 0
    labels = []
    predictions = []

    with tqdm(total=len(dataloader), desc="Validation") as pbar:
        for char_with_label, _ in dataloader:
            char = char_with_label[:, :, :-1]
            returns = char_with_label[:, :, -1]

            inputs = char.to(device).float()
            label = returns[:, -1].reshape(-1, 1).to(device).float()

            loss, rec, _, _, _, _ = factor_model(inputs, label)

            total_loss += loss.item() * inputs.size(0)
            pbar.update(1)

            predictions.append(rec.cpu().numpy())
            labels.append(label.cpu().numpy())

    preds_flat = np.concatenate(predictions, axis=0)
    labels_flat = np.concatenate(labels, axis=0)

    metrics = {
        'MSE': np.mean((preds_flat - labels_flat) ** 2),
        'MAE': np.mean(np.abs(preds_flat - labels_flat)),
        'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
        'R2': r2_score(preds_flat, labels_flat),
    }

    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics, predictions, labels


@torch.no_grad()
def test(factor_model, dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_model.to(device)
    factor_model.eval()
    total_loss = 0
    labels = []
    predictions = []
    with tqdm(total=len(dataloader), desc="Validation") as pbar:
        for char_with_label, _  in dataloader:
            char = char_with_label[:,:,:-1]
            returns = char_with_label[:,:,-1]
            
            inputs = char.to(device)
            label = returns[:,-1].reshape(-1,1).to(device)
            inputs = inputs.float()
            label = label.float()
            
            loss, rec, _, _, _, _ = factor_model(inputs, label)

            total_loss += loss.item() * inputs.size(0)
            pbar.update(1)

            predictions.append(rec.cpu().numpy())
            labels.append(label.cpu().numpy())

        preds_flat = np.concatenate(predictions, axis=0)
        labels_flat = np.concatenate(labels, axis=0)

        metrics = {
            'MSE': np.mean((preds_flat - labels_flat) ** 2),
            'MAE': np.mean(np.abs(preds_flat - labels_flat)),
            'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
            'R2': r2_score(preds_flat, labels_flat),
        }

        avg_loss = total_loss / len(dataloader)
        return avg_loss, metrics, predictions, labels
