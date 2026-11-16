import os
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# ================= 配置区域 =================
# 填入两个文件夹的路径
FOLDER_A = './Annotations'  # 原作者的 XML 文件夹
FOLDER_B = './annotation_second_person'  # 验证者（新标）的 XML 文件夹
IOU_THRESHOLD = 0.5  # IoU 阈值，通常取 0.5
BACKGROUND_CLASS = 'background'  # 用于表示漏标/多标的类别


# ===========================================

def parse_xml(xml_file):
    """解析 XML 获取 (xmin, ymin, xmax, ymax, class_name)"""
    if not os.path.exists(xml_file):
        return []

    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append({'bbox': [xmin, ymin, xmax, ymax], 'name': name})
    return objects


def calculate_iou(boxA, boxB):
    """计算两个框的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def compute_agreement():
    # 获取两个文件夹中的共同文件名
    files_a = set(f for f in os.listdir(FOLDER_A) if f.endswith('.xml'))
    files_b = set(f for f in os.listdir(FOLDER_B) if f.endswith('.xml'))
    common_files = list(files_a & files_b)

    print(f"Total files found in A: {len(files_a)}")
    print(f"Total files found in B: {len(files_b)}")
    print(f"Processing {len(common_files)} overlapping files for validation...")

    y_true = []  # Annotator A (Ground Truth)
    y_pred = []  # Annotator B (Validator)

    # 收集所有出现的类别
    all_classes = set([BACKGROUND_CLASS])

    for filename in common_files:
        path_a = os.path.join(FOLDER_A, filename)
        path_b = os.path.join(FOLDER_B, filename)

        objs_a = parse_xml(path_a)
        objs_b = parse_xml(path_b)

        # 记录每张图里已经被匹配过的索引，防止重复匹配
        matched_b_indices = set()

        # 1. 遍历 A 中的框，去 B 中找最佳匹配
        for i, obj_a in enumerate(objs_a):
            all_classes.add(obj_a['name'])
            best_iou = 0
            best_match_idx = -1

            for j, obj_b in enumerate(objs_b):
                if j in matched_b_indices: continue

                iou = calculate_iou(obj_a['bbox'], obj_b['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = j

            # 判断是否匹配成功
            if best_iou > IOU_THRESHOLD:
                # 匹配上了：对比类别
                obj_b = objs_b[best_match_idx]
                y_true.append(obj_a['name'])
                y_pred.append(obj_b['name'])
                matched_b_indices.add(best_match_idx)
                all_classes.add(obj_b['name'])
            else:
                # A 有框，B 没匹配上 -> B 视为 Background (漏检)
                y_true.append(obj_a['name'])
                y_pred.append(BACKGROUND_CLASS)

        # 2. 检查 B 中剩余没被匹配的框 (False Positives)
        for j, obj_b in enumerate(objs_b):
            if j not in matched_b_indices:
                all_classes.add(obj_b['name'])
                # B 有框，A 没匹配上 -> A 视为 Background
                y_true.append(BACKGROUND_CLASS)
                y_pred.append(obj_b['name'])

    # 计算 Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    print("-" * 30)
    print(f"Results on {len(common_files)} images:")
    print(f"Total Object Instances Compared: {len(y_true)}")
    print(f"Cohen's Kappa Coefficient: {kappa:.4f}")
    print("-" * 30)

    if kappa > 0.8:
        print("Conclusion: Excellent Agreement (Perfect for paper!)")
    elif kappa > 0.6:
        print("Conclusion: Substantial Agreement (Acceptable)")
    else:
        print("Conclusion: Low Agreement (Need to review defect definitions)")

    # 打印简易混淆矩阵看分布
    labels = sorted(list(all_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix (Rows=Annotator A, Cols=Annotator B):")
    print(f"{'':15} " + " ".join([f"{l[:6]:>6}" for l in labels]))
    for i, row in enumerate(cm):
        print(f"{labels[i]:15} " + " ".join([f"{val:6d}" for val in row]))


if __name__ == "__main__":
    compute_agreement()