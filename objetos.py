import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

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
    filename = f"objetos-{i}.jpg"
    path = os.path.join(folder, filename)

    img = cv2.imread(path)
    if img is None:
        print(f"Imagem não carregada: {filename}")
        continue

    print(f"\nProcessando: {filename}")

    resized = cv2.resize(img, (512, 512))  # Redimensionamento

    blurred = cv2.GaussianBlur(resized, (5, 5), 0)  # Desfoque
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Filtro Sharpening
    sharpened = cv2.filter2D(blurred, -1, kernel_sharpening)
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)  # Conversão escala de cinza
    equalized = cv2.equalizeHist(gray)  # Equalização de histograma

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

    # Armazenamento
    processed_images.append(equalized)

    # Visualização em uma só janela
    titles = ["Original", "Equalizada", "Threshold", "Contornos", "(Canny)",
              "Keypoints ORB", "Rotação", "Escalada", "Translação"]
    images = [resized, equalized, thresh, img_contours, edges,
              img_keypoints, rotated, scaled, translated]

    show_process_pipeline(titles, images)

print("\n=== PROCESSAMENTO FINALIZADO ===")
print(f"Total de imagens processadas: {len(processed_images)}")

# ==============================
# TREINAMENTO DE MODELO CNN (IA)
# ==============================
# Carregar CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Criar modelo CNN simples
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar modelo
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Salvar modelo
model.save('cnn_cifar10_model.h5')

# =================================
# CLASSIFICAÇÃO DE IMAGENS EXTERNAS
# =================================
model = load_model('cnn_cifar10_model.h5')

classes = ['avião', 'carro', 'pássaro', 'gato', 'veado',
           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

folder_classify = 'classify_images'

for filename in os.listdir(folder_classify):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(folder_classify, filename)

        # ================================
        # Carregamento e Pré-processamento
        # ================================
        img = cv2.imread(path)
        if img is None:
            print(f"Imagem não carregada: {filename}")
            continue

        img_resized = cv2.resize(img, (32, 32))  # Redimensionamento
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype('float32') / 255.0  # Normalização

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
