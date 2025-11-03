import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from detector import FaceDetector, non_max_suppression
import time
import fix_images

def draw_boxes(image_path, detections, output_path):
    """
    Desenha bounding boxes na imagem.
    
    Args:
        image_path: caminho da imagem original
        detections: lista de [x, y, w, h, confidence]
        output_path: caminho para salvar imagem com detecções
    """
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Tenta carregar uma fonte, senão usa a padrão
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Desenha cada detecção
    for det in detections:
        x, y, w, h, conf = det
        
        # Desenha retângulo
        draw.rectangle([x, y, x + w, y + h], outline='green', width=3)
        
        # Desenha label com confiança
        label = f'Face {conf:.2f}'
        
        # Fundo do texto
        bbox = draw.textbbox((x, y - 20), label, font=font)
        draw.rectangle(bbox, fill='green')
        draw.text((x, y - 20), label, fill='white', font=font)
    
    # Salva imagem
    image.save(output_path)
    print(f'Saved detection result to {output_path}')


def process_images(test_dir, output_dir, model_path='best_model.pth', 
                   confidence_threshold=0.7, iou_threshold=0.3,
                   window_size=64, stride=16):
    """
    Processa todas as imagens de um diretório.
    
    Args:
        test_dir: diretório com imagens de teste
        output_dir: diretório para salvar resultados
        model_path: caminho do modelo treinado
        confidence_threshold: threshold de confiança
        iou_threshold: threshold de IoU para NMS
        window_size: tamanho da janela de detecção
        stride: passo do sliding window
    """
    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializa detector
    print(f'Loading model from {model_path}...')
    detector = FaceDetector(model_path, confidence_threshold)
    print('Model loaded successfully!')
    
    # Processa todas as subpastas
    total_images = 0
    total_detections = 0
    
    for root, dirs, files in os.walk(test_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.pgm')):
                image_path = os.path.join(root, filename)
                
                print(f'\nProcessing: {filename}')
                t0 = time.time()
                
                # Detecta faces
                detections = detector.detect(image_path, window_size=window_size, stride=stride)
                print(f'  Found {len(detections)} raw detections')
                
                # Aplica NMS
                if len(detections) > 0:
                    detections_nms = non_max_suppression(detections, iou_threshold)
                    print(f'  After NMS: {len(detections_nms)} detections')
                else:
                    detections_nms = []
                
                # Desenha boxes e salva
                output_filename = f'detected_{filename}'
                output_path = os.path.join(output_dir, output_filename)
                draw_boxes(image_path, detections_nms, output_path)
                
                elapsed = time.time() - t0
                print(f'  Processed in {elapsed:.2f}s')
                
                total_images += 1
                total_detections += len(detections_nms)
    
    print(f'\n' + '='*60)
    print(f'Summary:')
    print(f'  Total images processed: {total_images}')
    print(f'  Total faces detected: {total_detections}')
    print(f'  Average detections per image: {total_detections/total_images:.2f}' if total_images > 0 else 0)
    print(f'  Results saved to: {output_dir}')
    print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Face detection with NMS')
    parser.add_argument('--test-dir', type=str, default='./test_images',
                       help='Directory with test images')
    parser.add_argument('--output-dir', type=str, default='./detection_results',
                       help='Directory to save detection results')
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.3,
                       help='IoU threshold for NMS (0-1)')
    parser.add_argument('--window-size', type=int, default=64,
                       help='Detection window size')
    parser.add_argument('--stride', type=int, default=16,
                       help='Sliding window stride')
    
    args = parser.parse_args()
    
    # Verifica se diretório de teste existe
    if not os.path.exists(args.test_dir):
        print(f'Error: Test directory "{args.test_dir}" not found!')
        return
    
    # Verifica se modelo existe
    if not os.path.exists(args.model_path):
        print(f'Error: Model file "{args.model_path}" not found!')
        print('Please train the model first using load_data.py')
        return
    
    # Processa imagens
    process_images(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        window_size=args.window_size,
        stride=args.stride
    )


if __name__ == '__main__':
    main()
