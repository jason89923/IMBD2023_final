from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import shutil

def load_images(folder_paths):
    image_size = (224, 224)
    images = []
    labels = []  # 用來存儲標籤信息
    image_paths = []  # 用來存儲圖片路徑
    for folder_path in folder_paths:
        label = folder_path.split('/')[-1]  # 從文件夾名稱中提取標籤
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, file_name)
                img = load_img(img_path, target_size=image_size)
                img = img_to_array(img)
                img = img / 255.0
                images.append(img)
                labels.append(label)
                image_paths.append(img_path)  # 存儲圖片的完整路徑
    return np.array(images), labels, image_paths  # 返回圖片、標籤和路徑

def kmeans_clustering(folder_paths, cluster_prefixes):
    images, labels, image_paths = load_images(folder_paths)  # 接收返回的路徑
    flattened_images = images.reshape(images.shape[0], -1)

    pca = PCA(n_components=50)
    flattened_images_pca = pca.fit_transform(flattened_images)

    centers_file_path = 'kmeans_centers.npy'
    if os.path.exists(centers_file_path):
        cluster_centers = np.load(centers_file_path)
    else:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        kmeans.fit(flattened_images_pca)
        np.save(centers_file_path, kmeans.cluster_centers_)
        cluster_centers = kmeans.cluster_centers_

    kmeans = KMeans(n_clusters=2, init=cluster_centers, n_init=1, max_iter=300)
    clusters = kmeans.fit_predict(flattened_images_pca)

    for i in range(2):
        for prefix in cluster_prefixes:
            cluster_folder = os.path.join('Project_B_training', f'cluster_{i}', prefix)
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)

    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        if img_path.endswith(('.png', '.jpg', '.jpeg')):
            final_cluster_label = clusters[i]
            final_target_folder = os.path.join('Project_B_training', f'cluster_{final_cluster_label}', label)
            if not os.path.exists(final_target_folder):
                os.makedirs(final_target_folder)  # 確保目標資料夾存在
            final_target_path = os.path.join(final_target_folder, os.path.basename(img_path))
            try:
                shutil.copy(img_path, final_target_path)
            except FileNotFoundError as e:
                print(f"錯誤: 無法複製 {img_path} 到 {final_target_path}: {e}")

    print("完成分群並將圖片分類到最終群組資料夾。")


folder_paths = [
    '/TOPIC/ProjectB/B_traing1/reworkable',
    '/TOPIC/ProjectB/B_traing1/not reworkable',
    '/TOPIC/ProjectB/B_traing2/reworkable',
    '/TOPIC/ProjectB/B_traing2/not reworkable'
]

cluster_prefixes = ['reworkable', 'not_reworkable']

kmeans_clustering(folder_paths, cluster_prefixes)
