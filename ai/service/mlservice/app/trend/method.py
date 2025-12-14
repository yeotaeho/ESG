"""학습 및 평가 메서드 계층"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
import logging

from .exceptions import TrainingError

logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: str) -> Dict:
    """
    한 에포크 학습
    
    Args:
        model: 학습할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 디바이스 ('cpu' 또는 'cuda')
        
    Returns:
        학습 메트릭 딕셔너리
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # 순전파
        optimizer.zero_grad()
        output = model(data)
        
        # 손실 계산
        if output.dim() == 1:
            loss = criterion(output, target.float())
        else:
            loss = criterion(output, target)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 정확도 계산 (분류 작업인 경우)
        if output.dim() > 1 and output.size(1) > 1:
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   device: str, task_type: str = 'classification') -> Dict:
    """
    모델 평가
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        criterion: 손실 함수
        device: 디바이스
        task_type: 작업 타입 ('classification' 또는 'regression')
        
    Returns:
        평가 메트릭 딕셔너리
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 손실 계산
            if output.dim() == 1:
                loss = criterion(output, target.float())
            else:
                loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # 예측값 수집
            if task_type == 'classification' and output.dim() > 1:
                pred = output.argmax(dim=1).cpu().numpy()
            else:
                pred = output.cpu().numpy().flatten()
            
            all_preds.extend(pred)
            all_targets.extend(target.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    
    # 메트릭 계산
    metrics = {
        'loss': avg_loss
    }
    
    if task_type == 'classification':
        metrics.update(calculate_classification_metrics(all_targets, all_preds))
    else:
        metrics.update(calculate_regression_metrics(all_targets, all_preds))
    
    return metrics


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     task_type: str = 'classification') -> Dict:
    """
    메트릭 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        task_type: 작업 타입 ('classification' 또는 'regression')
        
    Returns:
        메트릭 딕셔너리
    """
    if task_type == 'classification':
        return calculate_classification_metrics(y_true, y_pred)
    else:
        return calculate_regression_metrics(y_true, y_pred)


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    분류 메트릭 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        
    Returns:
        분류 메트릭 딕셔너리
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
    except Exception as e:
        logger.error(f"분류 메트릭 계산 실패: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    회귀 메트릭 계산
    
    Args:
        y_true: 실제값
        y_pred: 예측값
        
    Returns:
        회귀 메트릭 딕셔너리
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2)
        }
    except Exception as e:
        logger.error(f"회귀 메트릭 계산 실패: {e}")
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'r2_score': 0.0
        }


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """혼동 행렬 계산"""
    return confusion_matrix(y_true, y_pred)


def create_optimizer(model: nn.Module, optimizer_type: str = 'adam', lr: float = 0.001, **kwargs) -> optim.Optimizer:
    """
    옵티마이저 생성
    
    Args:
        model: 모델
        optimizer_type: 옵티마이저 타입 ('adam', 'sgd', 'rmsprop')
        lr: 학습률
        **kwargs: 추가 옵티마이저 파라미터
        
    Returns:
        옵티마이저
    """
    if optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, **kwargs)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    else:
        logger.warning(f"알 수 없는 옵티마이저 타입: {optimizer_type}, Adam 사용")
        return optim.Adam(model.parameters(), lr=lr)


def create_scheduler(optimizer: optim.Optimizer, scheduler_type: str = 'step', **kwargs) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    학습률 스케줄러 생성
    
    Args:
        optimizer: 옵티마이저
        scheduler_type: 스케줄러 타입 ('step', 'cosine', 'plateau')
        **kwargs: 스케줄러 파라미터
        
    Returns:
        학습률 스케줄러 (또는 None)
    """
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 100)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    elif scheduler_type == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    
    else:
        return None


class EarlyStopping:
    """조기 종료 클래스"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: 개선이 없을 때 기다릴 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'min' (손실 최소화) 또는 'max' (점수 최대화)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        조기 종료 체크
        
        Args:
            score: 현재 점수 (손실 또는 메트릭)
            
        Returns:
            조기 종료 여부
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """현재 점수가 더 나은지 확인"""
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader],
                config: Dict) -> Dict:
    """
    모델 학습
    
    Args:
        model: 학습할 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더 (선택사항)
        config: 학습 설정
            - epochs: 에포크 수
            - criterion: 손실 함수 타입 ('mse', 'crossentropy')
            - optimizer: 옵티마이저 타입 ('adam', 'sgd')
            - lr: 학습률
            - device: 디바이스
            - task_type: 작업 타입 ('classification', 'regression')
            - early_stopping: 조기 종료 설정 (선택사항)
            - scheduler: 스케줄러 설정 (선택사항)
            
    Returns:
        학습 결과 딕셔너리
    """
    try:
        device = config.get('device', 'cpu')
        epochs = config.get('epochs', 10)
        task_type = config.get('task_type', 'classification')
        
        # 손실 함수 설정
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss() if config.get('criterion') == 'crossentropy' else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # 옵티마이저 설정
        optimizer = create_optimizer(
            model,
            optimizer_type=config.get('optimizer', 'adam'),
            lr=config.get('lr', 0.001),
            **config.get('optimizer_kwargs', {})
        )
        
        # 스케줄러 설정
        scheduler = None
        if 'scheduler' in config:
            scheduler = create_scheduler(optimizer, **config['scheduler'])
        
        # 조기 종료 설정
        early_stopping = None
        if 'early_stopping' in config:
            es_config = config['early_stopping']
            early_stopping = EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0.0),
                mode=es_config.get('mode', 'min')
            )
        
        model.to(device)
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_metrics': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 학습
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
            
            # 검증
            val_metrics = {}
            if val_loader is not None:
                val_metrics = evaluate_model(model, val_loader, criterion, device, task_type)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
                history['val_metrics'].append(val_metrics)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['loss']:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics.get('accuracy', 0.0):.4f}")
                
                # 조기 종료 체크
                if early_stopping:
                    stop_metric = val_metrics['loss'] if task_type == 'regression' else -val_metrics.get('accuracy', 0.0)
                    if early_stopping(stop_metric):
                        logger.info(f"조기 종료: Epoch {epoch+1}")
                        break
                
                # 최고 모델 저장
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # 스케줄러 업데이트
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and val_loader:
                    scheduler.step(val_metrics['loss'])
                elif not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
        
        return {
            'history': history,
            'best_val_loss': best_val_loss,
            'final_val_metrics': val_metrics if val_loader else {}
        }
        
    except Exception as e:
        logger.error(f"모델 학습 실패: {e}")
        raise TrainingError(f"모델 학습 실패: {e}")


def create_dataloader(X, y, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    데이터 로더 생성
    
    Args:
        X: 특성 데이터
        y: 타겟 데이터
        batch_size: 배치 크기
        shuffle: 셔플 여부
        
    Returns:
        데이터 로더
    """
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if isinstance(y, np.ndarray):
        y = torch.LongTensor(y) if y.dtype in [np.int64, np.int32] else torch.FloatTensor(y)
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

