import random
import os


def generate_split_file(total_images=1065, seed=42):
    # 1. 准备文件名列表 (0.jpg 到 1064.jpg)
    filenames = [f"{i}.jpg" for i in range(total_images)]

    # 2. 设置随机种子以保证可复现性 (Reviewer 重点关注点)
    random.seed(seed)

    # 3. 打乱列表
    shuffled_files = filenames.copy()
    random.shuffle(shuffled_files)

    # 4. 计算划分数量 (70% / 15% / 15%)
    n_train = int(total_images * 0.70)
    n_val = int(total_images * 0.15)
    # 剩余归为测试集

    train_files = set(shuffled_files[:n_train])
    val_files = set(shuffled_files[n_train:n_train + n_val])
    test_files = set(shuffled_files[n_train + n_val:])

    # 5. 写入文件
    # 为了方便用户查阅，我们按文件名顺序(0->1064)写入，并在后面标注它属于哪个集
    output_filename = "train_val_test_split.txt"
    with open(output_filename, "w") as f:
        # 写入表头
        f.write("ImageID,Subset\n")

        for i in range(total_images):
            fname = f"{i}.jpg"
            if fname in train_files:
                subset = "train"
            elif fname in val_files:
                subset = "val"
            else:
                subset = "test"

            f.write(f"{fname},{subset}\n")

    print(f"Successfully generated {output_filename}")
    print(f"Stats: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


if __name__ == "__main__":
    generate_split_file()