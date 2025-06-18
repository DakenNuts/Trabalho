import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# ====================
# EXIBIÇÃO DAS IMAGENS
# ====================
def show_process_pipeline(titles, images):
    plt.figure(figsize=(5 * len(images), 10))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(2, len(images) // 2 + len(images) % 2, i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# =========================
# PROCESSAMENTO DAS IMAGENS
# =========================
folder = "images"
total_images = 10

processed_images = []
keypoints_list = []
descriptors_list = []

for i in range(1, total_images + 1):
    filename = f"foto-{i}.jpg"
    path = os.path.join(folder, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Imagem não carregada: {filename}")
        continue

    print(f"\nProcessando: {filename}")

    resized = cv2.resize(img, (512, 512)) # Redimensionamento

    blurred = cv2.GaussianBlur(resized, (5, 5), 0) # Desfoque
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) # Filtro Sharpening
    sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY) # Conversão escala de cinza
    equalized = cv2.equalizeHist(gray) # Equalização de histograma

    # ======================
    # Detecção e Segmentação
    # ======================
    _, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Thresholding
    edges = cv2.Canny(equalized, 100, 200)  # Canny

    # Contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

    # ===========================
    # Extração de Características
    # ===========================
    orb = cv2.ORB_create()  # ORB
    keypoints, descriptors = orb.detectAndCompute(equalized, None)  # Detecção de Keypoints
    img_keypoints = cv2.drawKeypoints(equalized, keypoints, None, color=(0, 255, 0))  # Visualização de Keypoints

    print(f"Keypoints detectados: {len(keypoints)}")

    keypoints_list.append(keypoints)
    if descriptors is not None:
        descriptors_list.append(descriptors)
    else:
        descriptors_list.append(np.zeros((1, 32)))

    # ==========================
    # Transformações Geométricas
    # ==========================
    rows, cols = equalized.shape
    # Rotação
    M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)  # Rotação
    rotated = cv2.warpAffine(equalized, M_rotate, (cols, rows))
    # Escala
    scaled = cv2.resize(equalized, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    # Translação
    M_translate = np.float32([[1, 0, 50], [0, 1, 30]])
    translated = cv2.warpAffine(equalized, M_translate, (cols, rows))

    processed_images.append(equalized)

    titles = ["Original", "Equalizada", "Threshold", "Contornos", "(Canny)",
              "Keypoints ORB", "Rotação", "Escalada", "Translação"]
    images = [resized, equalized, thresh, img_contours, edges,
              img_keypoints, rotated, scaled, translated]

    show_process_pipeline(titles, images)

keypoints_counts = [len(kp) for kp in keypoints_list]
image_names = [f"foto-{i}.jpg" for i in range(1, len(keypoints_list) + 1)]

plt.figure(figsize=(10, 6))
plt.barh(image_names, keypoints_counts, color='skyblue')
plt.xlabel("Quantidade de Keypoints")
plt.ylabel("Imagens")
plt.title("Keypoints Detectados por Imagem")
plt.tight_layout()
plt.show()

print("\n=== PROCESSAMENTO FINALIZADO ===")
print(f"Total de imagens processadas: {len(processed_images)}")

# ==============================
# TREINAMENTO DE MODELO CNN (IA)
# ==============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# ================================
# GRÁFICO DE DESEMPENHO DO MODELO
# ================================
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.show()

# ================================
# MATRIZ DE CONFUSÃO DO CIFAR-10
# ================================
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
classes = ['avião', 'carro', 'pássaro', 'gato', 'veado',
           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - CIFAR10')
plt.tight_layout()
plt.show()

# Salvar modelo
model.save('cnn_cifar10_model.h5')

# =================================
# CLASSIFICAÇÃO DE IMAGENS EXTERNAS
# =================================
model = load_model('cnn_cifar10_model.h5')

folder_images = 'images'
results = []

for filename in os.listdir(folder_images):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(folder_images, filename)

        # ================================
        # Carregamento e Pré-processamento
        # ================================
        img = cv2.imread(path)
        if img is None:
            print(f"Imagem não carregada: {filename}")
            continue

        img_resized = cv2.resize(img, (32, 32))  # Redimensionamento
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype('float32') / 255.0

        img_input = np.expand_dims(img_normalized, axis=0)

        # ========
        # Predição
        # ========
        pred = model.predict(img_input)
        class_id = np.argmax(pred)
        confidence = pred[0][class_id]

        # =========
        # Resultado
        # =========
        print(f"\nImagem: {filename}")
        print(f"Classe prevista: {classes[class_id]}")
        print(f"Confiança: {confidence:.2f}")

        results.append({
            'Imagem': filename,
            'Classe Prevista': classes[class_id],
            'Confiança (%)': f"{confidence * 100:.2f}"
        })

# ===========================
# TABELA DE RESULTADOS FINAIS
# ===========================
df_results = pd.DataFrame(results)
print("\n=== TABELA DE RESULTADOS ===")
print(df_results)

plt.figure(figsize=(8, len(df_results) * 0.6))
plt.axis('off')
table = plt.table(cellText=df_results.values,
                   colLabels=df_results.columns,
                   cellLoc='center',
                   loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('Classificação de Imagens Externas')
plt.show()
