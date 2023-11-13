from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# 設定資料夾路徑和圖片尺寸
folder_path = 'Project_B/reworkable_img'  # 更換為您圖片資料夾的路徑
image_size = (64, 64)  # 更換為適合您模型的尺寸

# 加載圖片並預處理
def load_images(folder_path, image_size):
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 檢查檔案格式
            img_path = os.path.join(folder_path, file_name)
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img)
            img = img / 255.0  # 歸一化到 [0, 1]
            images.append(img)
    return np.array(images)

# 加載數據集
images = load_images(folder_path, image_size)

# 展平圖片
flattened_images = images.reshape(images.shape[0], -1)

# 特徵提取，使用 PCA 降維
pca = PCA(n_components=50)  # 可以根據需要調整組件數
flattened_images_pca = pca.fit_transform(flattened_images)

# 執行 K-Means 分群
k = 10  # 指定群組數量
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(flattened_images_pca)

# 現在 clusters 包含每張圖片對應的群組標籤
