import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO # Para a detecção de objetos
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans

# 1. CARREGAR OS MODELOS
# YOLOv11 pré-treinado (detecta 'bird' nativamente)
yolo_model = YOLO('yolo11n.pt') 

# Seu modelo MobileNetV2 treinado na FATEC
class_model = tf.keras.models.load_model('modelo_guara_mobilenetv2.keras')
labels = ['colhereiro', 'garca', 'guara_vermelho']

def pipeline_monitoramento(image_path):
    # 1. SEMPRE carregue uma cópia "fresca" do disco
    # Isso garante que cada execução comece do zero
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return "Erro ao carregar"
    
    img_rgb_para_cor = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # ESTA FICA PURA
    img_yolo = img_bgr.copy() # ESTA VAI PRO YOLO
    
    # --- ETAPA 1: YOLO ---
    results = yolo_model(img_yolo, verbose=False)
    
    deteccao_final = []
    for r in results:
        for box in r.boxes:
            if yolo_model.names[int(box.cls)] == 'bird':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # --- ETAPA 2: RECORTE SEGURO ---
                # Pegamos o recorte da imagem "PURA" (0-255)
                passaro_raw = img_rgb_para_cor[y1:y2, x1:x2].copy()
                
                # --- ETAPA 3: CLASSIFICAÇÃO (AQUI PODE "ESTRAGAR") ---
                # Criamos uma versão só para o modelo MobileNetV2
                passaro_input = cv2.resize(passaro_raw, (224, 224))
                passaro_input = np.expand_dims(passaro_input, axis=0)
                
                # O preprocess_input SÓ toca na variável 'passaro_input'
                passaro_input = preprocess_input(passaro_input) 
                
                preds = class_model.predict(passaro_input, verbose=0)
                especie = labels[np.argmax(preds)]

                # --- ETAPA 4: K-MEANS NO 'RAW' ---
                cor_dominante = None
                if especie == 'guara_vermelho':
                    # Usamos o 'passaro_raw' que NUNCA viu o preprocess_input
                    # IMPORTANTE: Força uint8 e cópia profunda para evitar estado compartilhado
                    passaro_raw_safe = np.uint8(passaro_raw.copy())
                    pixels = cv2.resize(passaro_raw_safe, (50, 50)).reshape(-1, 3).astype(np.float32)
                    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(pixels)
                    cor_dominante = np.uint8(kmeans.cluster_centers_[0]).tolist()

                deteccao_final.append({
                    "especie": especie,
                    "cor_rgb": cor_dominante
                })

    return deteccao_final

# Exemplo de uso:
resultados = pipeline_monitoramento('guara4.jpg')
print(resultados)