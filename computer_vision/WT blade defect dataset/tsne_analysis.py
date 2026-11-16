import os
import sys


import matplotlib


matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
# ========================================================

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from skimage.feature import hog, local_binary_pattern
from sklearn.manifold import TSNE
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ================= 配置路径 =================
IMG_DIR = r'./JPEGImages'
XML_DIR = r'./Annotations'
RESIZE_W, RESIZE_H = 64, 128


def extract_features(img_dir, xml_dir):
    features_hog = []
    features_lbp = []
    labels = []

    # 检查路径是否存在
    if not os.path.exists(xml_dir):
        print(f"错误：找不到路径 {xml_dir}")
        return [], [], []

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"正在处理 {len(xml_files)} 个XML文件...")

    valid_count = 0
    for xml_file in xml_files:
        try:
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            filename_node = root.find('filename').text
            base_name = os.path.splitext(filename_node)[0]

            img_path = None
            for ext in ['.jpg', '.JPG', '.png', '.jpeg']:
                temp_path = os.path.join(img_dir, base_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break

            if img_path is None: continue

            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                bndbox = obj.find('bndbox')

                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                roi = gray[max(0, ymin):min(gray.shape[0], ymax),
                      max(0, xmin):min(gray.shape[1], xmax)]

                if roi.size == 0: continue
                roi_resized = cv2.resize(roi, (RESIZE_W, RESIZE_H))

                # HOG
                fd_hog = hog(roi_resized, orientations=9, pixels_per_cell=(16, 16),
                             cells_per_block=(2, 2), visualize=False)

                # LBP
                radius = 3
                n_points = 8 * radius
                lbp = local_binary_pattern(roi_resized, n_points, radius, method='uniform')
                (hist, _) = np.histogram(lbp.ravel(),
                                         bins=np.arange(0, n_points + 3),
                                         range=(0, n_points + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)

                features_hog.append(fd_hog)
                features_lbp.append(hist)
                labels.append(cls_name)
                valid_count += 1

        except Exception:
            continue

    print(f"成功提取特征: {valid_count} 个")
    return np.array(features_hog), np.array(features_lbp), np.array(labels)


def plot_combined_tsne(X_hog, X_lbp, y, save_name="Fig6_Feature_Visualization.png"):
    print("正在计算 t-SNE...")

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    hog_embedded = tsne.fit_transform(X_hog)
    lbp_embedded = tsne.fit_transform(X_lbp)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    unique_labels = np.unique(y)
    palette = sns.color_palette("bright", len(unique_labels))

    # HOG
    df_hog = pd.DataFrame(hog_embedded, columns=['Dim1', 'Dim2'])
    df_hog['Class'] = y
    sns.scatterplot(data=df_hog, x='Dim1', y='Dim2', hue='Class', style='Class',
                    palette=palette, ax=axes[0], s=80, alpha=0.8)
    axes[0].set_title('(a) t-SNE of HOG Features')
    axes[0].legend_.remove()

    # LBP
    df_lbp = pd.DataFrame(lbp_embedded, columns=['Dim1', 'Dim2'])
    df_lbp['Class'] = y
    sns.scatterplot(data=df_lbp, x='Dim1', y='Dim2', hue='Class', style='Class',
                    palette=palette, ax=axes[1], s=80, alpha=0.8)
    axes[1].set_title('(b) t-SNE of LBP Features')
    axes[1].legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(unique_labels))

    plt.tight_layout()

    print(f"正在保存图片到当前目录: {save_name}")
    # 注意：这里不需要 plt.show()，Agg模式下 show() 会报错或无效
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print("✅ 保存成功！请在文件夹中查看图片。")


if __name__ == "__main__":
    feat_hog, feat_lbp, labels = extract_features(IMG_DIR, XML_DIR)
    if len(labels) > 0:
        plot_combined_tsne(feat_hog, feat_lbp, labels)
    else:
        print("未检测到数据，请检查路径。")