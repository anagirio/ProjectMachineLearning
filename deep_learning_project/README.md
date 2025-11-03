# Detector de Faces com NMS

Arquivos criados para adicionar detecção de faces com Non-Maximum Suppression ao seu projeto.

## Arquivos Criados

1. **fix_images.py** - Configuração para tolerar imagens truncadas/corrompidas
   - É importado automaticamente pelos outros arquivos

2. **detector.py** - Implementa:
   - Classe `FaceDetector`: detector com sliding window em múltiplas escalas
   - Função `non_max_suppression`: elimina detecções redundantes usando IoU

3. **detect_faces.py** - Script principal para:
   - Processar imagens do diretório de teste
   - Aplicar detecção + NMS
   - Salvar imagens com bounding boxes desenhadas
   - Suporta formatos: .png, .jpg, .jpeg, .bmp, .gif, .pgm

4. **Arquivos _fixed.py** - Versões corrigidas dos arquivos originais:
   - `load_data_fixed.py` - versão do load_data.py com correção de imagens
   - `test_fixed.py` - versão do test.py com correção de imagens

## IMPORTANTE: Correção de Imagens Truncadas

Se você encontrar o erro "OSError: image file is truncated":

**Solução Rápida:** Use os arquivos _fixed.py que já têm a correção:
```bash
python load_data_fixed.py --epochs 10
python test_fixed.py
```

**OU** adicione esta linha no início dos seus arquivos originais (load_data.py, test.py):
```python
import fix_images  # Logo após os outros imports
```

## Como Usar

### 1. Certifique-se de ter o modelo treinado

**Usando o arquivo corrigido (recomendado):**
```bash
python load_data_fixed.py --epochs 10
```

**OU** use o arquivo original (se já adicionou `import fix_images`):
```bash
python load_data.py --epochs 10
```

Isso vai gerar o arquivo `best_model.pth`

### 2. Execute a detecção

**Para imagens 36x36 pixels (como .pgm do projeto):**
```bash
python detect_faces.py --window-size 36 --stride 4 --confidence 0.5
```

**Para imagens maiores (tamanho padrão):**
```bash
python detect_faces.py
```

### 3. Parâmetros opcionais
```bash
python detect_faces.py \
    --test-dir ./test_images \
    --output-dir ./detection_results \
    --model-path best_model.pth \
    --confidence 0.5 \
    --iou 0.3 \
    --window-size 36 \
    --stride 4
```

## Parâmetros Explicados

- `--test-dir`: Diretório com as imagens para detectar faces (padrão: ./test_images)
- `--output-dir`: Onde salvar as imagens com detecções (padrão: ./detection_results)
- `--model-path`: Caminho do modelo treinado (padrão: best_model.pth)
- `--confidence`: Threshold de confiança (0-1). Valores maiores = menos detecções, mais precisas
- `--iou`: Threshold de IoU para NMS (0-1). Valores menores = mais agressivo na eliminação de duplicatas
- `--window-size`: Tamanho da janela de detecção em pixels (IMPORTANTE: deve corresponder ao tamanho das faces nas suas imagens)
- `--stride`: Passo do sliding window (menor = mais detecções, mais lento)

## ATENÇÃO: Tamanho da Janela (window-size)

O parâmetro `--window-size` é **CRÍTICO** e deve corresponder ao tamanho aproximado das faces nas suas imagens:

- **Imagens 36x36 pixels** (como .pgm do projeto): use `--window-size 36`
- **Imagens maiores com faces pequenas**: use `--window-size 64` ou `--window-size 48`
- **Imagens grandes com faces grandes**: use `--window-size 128` ou maior

**Se window-size for maior que a imagem, NÃO DETECTARÁ NADA!**

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
5. Imprimir resumo

**Exemplo de saída real:**
```
Loading model from best_model.pth...
Model loaded successfully!

Processing: face_00001.pgm
  Found 3 raw detections
  After NMS: 1 detections
  Processed in 0.12s
Saved detection result to ./detection_results/detected_face_00001.pgm

...

Summary:
  Total images processed: 7628
  Total faces detected: 2740
  Average detections per image: 0.36
  Results saved to: ./detection_results
```

## Ajuste Fino

### Se tiver **muitas detecções falsas** (falsos positivos):
```bash
# Aumentar confiança mínima
python detect_faces.py --window-size 36 --confidence 0.7

# NMS mais agressivo
python detect_faces.py --window-size 36 --iou 0.2

# Ambos
python detect_faces.py --window-size 36 --confidence 0.7 --iou 0.2
```

### Se **não detectar faces** (ou muito poucas):
```bash
# Diminuir confiança
python detect_faces.py --window-size 36 --confidence 0.3

# Stride menor (mais detecções)
python detect_faces.py --window-size 36 --stride 2 --confidence 0.4

# Verificar tamanho da janela
file test_images/1/* | head -1  # Ver tamanho real das imagens
python detect_faces.py --window-size [TAMANHO_CORRETO]
```

### Se estiver **muito lento**:
```bash
# Aumentar stride
python detect_faces.py --window-size 36 --stride 8

# Reduzir escalas (edite detector.py, linha: scales=[0.5, 0.75, 1.0, 1.25, 1.5])
# Mude para: scales=[1.0]
```

## Verificação de Imagens

Para ver o tamanho das suas imagens:
```bash
file test_images/0/* | head -3
file test_images/1/* | head -3
```

Ajuste o `--window-size` de acordo!

## Estrutura de Diretórios

```
projeto/
├── fix_images.py              ⬅️ OBRIGATÓRIO
├── net.py                     (seu arquivo original)
├── load_data.py               (seu arquivo original)
├── test.py                    (seu arquivo original)
├── detector.py                ⬅️ NOVO
├── detect_faces.py            ⬅️ NOVO
├── load_data_fixed.py         ⬅️ NOVO (opcional)
├── test_fixed.py              ⬅️ NOVO (opcional)
├── best_model.pth             (gerado pelo treinamento)
├── train_images/
│   ├── 0/                     (ou noface/)
│   └── 1/                     (ou face/)
├── test_images/
│   ├── 0/
│   └── 1/
└── detection_results/         (gerado automaticamente)
    ├── detected_image1.pgm
    ├── detected_image2.pgm
    └── ...
```

## Integração com Código Existente

Os novos arquivos funcionam com:
- `net.py`: sua arquitetura de rede
- `load_data.py`: usa o modelo treinado por ele
- `test.py`: pode continuar usando para avaliar acurácia

Não é necessário modificar os arquivos antigos (a menos que queira adicionar `import fix_images`)!

## Comandos Rápidos

```bash
# Treinar modelo
python load_data_fixed.py --epochs 10

# Testar acurácia
python test_fixed.py

# Detectar faces (imagens 36x36)
python detect_faces.py --window-size 36 --stride 4 --confidence 0.5

# Detectar com mais sensibilidade
python detect_faces.py --window-size 36 --stride 4 --confidence 0.3

# Detectar com mais precisão
python detect_faces.py --window-size 36 --stride 4 --confidence 0.7

# Ver ajuda completa
python detect_faces.py --help
```

## Troubleshooting

**Erro: "No module named 'fix_images'"**
- Certifique-se que fix_images.py está na mesma pasta

**Erro: "OSError: image file is truncated"**
- Use load_data_fixed.py ou adicione `import fix_images`

**Erro: "Total faces detected: 0"**
- Verifique o tamanho das imagens com `file test_images/1/* | head -1`
- Ajuste `--window-size` para o tamanho correto
- Tente `--confidence 0.3` para threshold mais baixo

**Detecção muito lenta**
- Aumente `--stride` (ex: 8 ou 16)
- Considere usar menos escalas (edite detector.py)

**Imagens de saída não aparecem**
- Verifique a pasta `detection_results/`
- Apenas imagens COM detecções são salvas

### Participants:
- Ana Luisa Girio
- Julia Guimarães Simão
- Leonardo
- Maria Isabel
- Nicolas
