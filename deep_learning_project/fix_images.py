"""
Configuração para tolerar imagens truncadas ou corrompidas.

Este módulo configura o PIL (Python Imaging Library) para tentar
carregar imagens mesmo que estejam incompletas ou parcialmente corrompidas.

Para usar, simplesmente importe este módulo no início de qualquer script
que trabalhe com imagens:

    import fix_images

Isso deve ser feito ANTES de qualquer operação de carregamento de imagens.
"""

from PIL import ImageFile

# Permite carregar imagens truncadas (incompletas)
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("PIL configured to load truncated images")
