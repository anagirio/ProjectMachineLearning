# Detector de Faces com NMS

Arquivos criados para adicionar detecção de faces com Non-Maximum Suppression ao seu projeto.

## Arquivos Criados

1. **detector.py** - Implementa:
   - Classe `FaceDetector`: detector com sliding window em múltiplas escalas
   - Função `non_max_suppression`: elimina detecções redundantes usando IoU

2. **detect_faces.py** - Script principal para:
   - Processar imagens do diretório de teste
   - Aplicar detecção + NMS
   - Salvar imagens com bounding boxes desenhadas

## Como Usar

### 1. Certifique-se de ter o modelo treinado
```bash
python load_data.py --epochs 10
```
Isso vai gerar o arquivo `best_model.pth`

### 2. Execute a detecção
```bash
python detect_faces.py
```

### 3. Parâmetros opcionais
```bash
python detect_faces.py \
    --test-dir ./test_images \
    --output-dir ./detection_results \
    --model-path best_model.pth \
    --confidence 0.7 \
    --iou 0.3 \
    --window-size 64 \
    --stride 16
```

## Parâmetros Explicados

- `--test-dir`: Diretório com as imagens para detectar faces
- `--output-dir`: Onde salvar as imagens com detecções
- `--model-path`: Caminho do modelo treinado
- `--confidence`: Threshold de confiança (0-1). Valores maiores = menos detecções, mais precisas
- `--iou`: Threshold de IoU para NMS (0-1). Valores menores = mais agressivo na eliminação de duplicatas
- `--window-size`: Tamanho da janela de detecção (pixels)
- `--stride`: Passo do sliding window (menor = mais detecções, mais lento)

## Como Funciona

### Sliding Window
O detector percorre cada imagem em múltiplas escalas (0.5x, 0.75x, 1.0x, 1.25x, 1.5x) com janelas de tamanho fixo:
- Isso permite detectar faces de diferentes tamanhos
- Cada janela é classificada pela sua rede neural
- Janelas com confiança > threshold são salvas como detecções

### Non-Maximum Suppression (NMS)
Elimina bounding boxes redundantes:
1. Ordena todas as detecções por confiança (maior primeiro)
2. Para cada detecção:
   - Mantém a de maior confiança
   - Remove todas as outras que têm IoU > threshold com ela
3. Resultado: apenas as melhores detecções não-sobrepostas

### IoU (Intersection over Union)
Métrica que mede sobreposição entre duas bounding boxes:
- IoU = Área de Interseção / Área de União
- IoU = 0: boxes não se tocam
- IoU = 1: boxes idênticas

## Saída Esperada

O script vai:
1. Processar cada imagem do `test_dir`
2. Detectar faces usando sliding window
3. Aplicar NMS para eliminar duplicatas
4. Salvar imagens com boxes verdes e labels de confiança
5. Imprimir resumo:
```
Processing: image001.jpg
  Found 15 raw detections
  After NMS: 2 detections
  Processed in 3.45s
Saved detection result to ./detection_results/detected_image001.jpg

Summary:
  Total images processed: 10
  Total faces detected: 18
  Average detections per image: 1.80
```

## Ajuste Fino

Se tiver **muitas detecções falsas**:
- Aumente `--confidence` (ex: 0.8 ou 0.9)
- Diminua `--iou` (ex: 0.2)

Se **não detectar faces**:
- Diminua `--confidence` (ex: 0.5 ou 0.6)
- Diminua `--stride` (ex: 8) para mais detecções
- Ajuste `--window-size` conforme tamanho das faces nas suas imagens

Se estiver **muito lento**:
- Aumente `--stride` (ex: 32)
- Reduza número de escalas editando `detector.py` (linha com `scales=[...]`)

## Integração com Código Existente

Os novos arquivos funcionam com:
- `net.py`: sua arquitetura de rede
- `load_data.py`: usa o modelo treinado por ele
- `test.py`: pode continuar usando para avaliar acurácia

Não é necessário modificar os arquivos antigos!
