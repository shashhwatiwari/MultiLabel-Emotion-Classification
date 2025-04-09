import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, jaccard_score
import torch
from torch import nn

def calculate_metrics(y_true, y_pred):
    """Calculate various metrics for multi-label classification"""
    metrics = {}
    
    # Convert to numpy arrays if they're tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Convert predictions to binary (0 or 1)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro')
    metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro')
    metrics['precision_micro'] = precision_score(y_true, y_pred_binary, average='micro')
    metrics['precision_macro'] = precision_score(y_true, y_pred_binary, average='macro')
    metrics['recall_micro'] = recall_score(y_true, y_pred_binary, average='micro')
    metrics['recall_macro'] = recall_score(y_true, y_pred_binary, average='macro')
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred_binary)
    metrics['jaccard_score'] = jaccard_score(y_true, y_pred_binary, average='samples')
    
    return metrics

def evaluate_model(model, data_loader, device, criterion=nn.BCEWithLogitsLoss()):
    """Evaluate model on a data loader"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_predictions.append(logits.sigmoid())
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics

def print_metrics(metrics, prefix=''):
    """Print metrics in a formatted way"""
    print(f"\n{prefix}Metrics:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"F1 Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (Micro): {metrics['precision_micro']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Micro): {metrics['recall_micro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Jaccard Score: {metrics['jaccard_score']:.4f}")

def save_metrics(metrics, file_path):
    """Save metrics to a file"""
    with open(file_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def calculate_metrics_single_label(y_true, y_pred):
    """Calculate metrics for single-label classification"""
    metrics = {}
    
    # Convert to numpy arrays if they're tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Get predicted class (argmax)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Calculate metrics
    metrics['f1_micro'] = f1_score(y_true_classes, y_pred_classes, average='micro')
    metrics['f1_macro'] = f1_score(y_true_classes, y_pred_classes, average='macro')
    metrics['precision_micro'] = precision_score(y_true_classes, y_pred_classes, average='micro')
    metrics['precision_macro'] = precision_score(y_true_classes, y_pred_classes, average='macro')
    metrics['recall_micro'] = recall_score(y_true_classes, y_pred_classes, average='micro')
    metrics['recall_macro'] = recall_score(y_true_classes, y_pred_classes, average='macro')
    metrics['accuracy'] = np.mean(y_true_classes == y_pred_classes)
    
    return metrics

def evaluate_model_single_label(model, data_loader, device, criterion=nn.CrossEntropyLoss()):
    """Evaluate model on a data loader for single-label classification"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_predictions.append(logits)
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics_single_label(all_labels, all_predictions)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics 