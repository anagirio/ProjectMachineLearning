import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from net import Net
import fix_images  

class FaceDetector:
    def __init__(self, model_path='best_model.pth', confidence_threshold=0.7):
        """
        Inicializa o detector de faces.
        
        Args:
            model_path: caminho do modelo treinado
            confidence_threshold: threshold mínimo de confiança para detectar face
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Net().to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        self.confidence_threshold = confidence_threshold
        
        # Transformação para as janelas (mesma do treinamento)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0,), std=(1,))
        ])
    
    def detect(self, image_path, window_size=64, stride=16, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """
        Detecta faces em uma imagem usando sliding window em múltiplas escalas.
        
        Args:
            image_path: caminho da imagem
            window_size: tamanho da janela de detecção
            stride: passo do sliding window
            scales: escalas para buscar faces de diferentes tamanhos
            
        Returns:
            Lista de bounding boxes (x, y, w, h, confidence)
        """
        # Carrega imagem original
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        detections = []
        
        # Sliding window em múltiplas escalas
        for scale in scales:
            # Redimensiona a imagem
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            scaled_image = image.resize((new_width, new_height), Image.BILINEAR)
            
            # Percorre a imagem com sliding window
            for y in range(0, new_height - window_size, stride):
                for x in range(0, new_width - window_size, stride):
                    # Extrai janela
                    window = scaled_image.crop((x, y, x + window_size, y + window_size))
                    
                    # Classifica a janela
                    confidence = self._classify_window(window)
                    
                    # Se confiança acima do threshold, adiciona detecção
                    if confidence >= self.confidence_threshold:
                        # Converte coordenadas de volta para escala original
                        x_orig = int(x / scale)
                        y_orig = int(y / scale)
                        w_orig = int(window_size / scale)
                        h_orig = int(window_size / scale)
                        
                        detections.append([x_orig, y_orig, w_orig, h_orig, confidence])
        
        return detections
    
    def _classify_window(self, window):
        """
        Classifica uma janela da imagem.
        
        Args:
            window: janela PIL Image
            
        Returns:
            Confiança de que há uma face (0-1)
        """
        # Aplica transformação
        window_tensor = self.transform(window).unsqueeze(0).to(self.device)
        
        # Predição
        with torch.no_grad():
            output = self.net(window_tensor)
            probabilities = F.softmax(output, dim=1)
            # Retorna probabilidade da classe 'face' (índice 1)
            confidence = probabilities[0][1].item()
        
        return confidence


def non_max_suppression(detections, iou_threshold=0.3):
    """
    Aplica Non-Maximum Suppression para eliminar detecções redundantes.
    
    Args:
        detections: lista de [x, y, w, h, confidence]
        iou_threshold: threshold de IoU para considerar boxes como duplicadas
        
    Returns:
        Lista filtrada de detecções
    """
    if len(detections) == 0:
        return []
    
    # Converte para numpy array
    boxes = np.array(detections)
    
    # Extrai coordenadas e scores
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    scores = boxes[:, 4]
    
    # Calcula áreas
    areas = (x2 - x1) * (y2 - y1)
    
    # Ordena por score (decrescente)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # Pega o box com maior score
        i = order[0]
        keep.append(i)
        
        # Calcula IoU com os demais boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Mantém apenas boxes com IoU abaixo do threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep].tolist()
