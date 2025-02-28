from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
import shutil
import time

def calculate_position_error(pred_box, min_box):
    x_center_pred, y_center_pred, _, _ = change_center(pred_box, width, height)
    x_center_cor, y_center_cor, _, _ = change_center(min_box, width, height)

    position_error = np.sqrt((x_center_pred - x_center_cor)**2 + (y_center_pred - y_center_cor)**2)

    return position_error

def change_center(box, width, height):
    x_center = box[1] * width
    y_center = box[2] * height
    w = box[3] * width
    h = box[4] * height

    return x_center, y_center, w, h

def correct(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        class_label = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        boxes.append((class_label, x_center, y_center, width, height))
    return boxes

def accuracy(predictions, correct_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    matched_correct_boxes = set()  # 正解ボックスでマッチしたものを追跡
    total_predicitions = len(predictions)

    for pred in predictions:
        match = False
        pred_bbox = pred.xyxy[0].tolist()
        pred_class = int(pred.cls)

        for i, correct in enumerate(correct_boxes):
            cor = change_end(correct, width, height)
            correct_bbox = list(cor[0:4])
            correct_class = int(correct[0])
            iou = calculate_iou(pred_bbox, correct_bbox)

            if iou >= iou_threshold and pred_class == correct_class:
                if i not in matched_correct_boxes:  # 重複してマッチしないように
                    tp += 1
                    matched_correct_boxes.add(i)
                    match = True
                    break
        
        if not match:
            fp += 1  # 予測が正解ボックスにマッチしなかった場合は偽陽性

    # マッチしなかった正解ボックスは偽陰性
    fn = len(correct_boxes) - len(matched_correct_boxes)

    # 精度、再現率の計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if total_predicitions > 0:
        accu = tp / total_predicitions
    else:
        accu = 0

    return accu, precision, recall, tp, fp, fn


def calculate_iou(pred, correct):
    xa1, ya1, xa2, ya2 = pred
    xb1, yb1, xb2, yb2 = correct

    x1 = max(xa1, xb1)
    y1 = max(ya1, yb1)
    x2 = min(xa2, xb2)
    y2 = min(ya2, yb2)

    inter = max(0, x2-x1) * max(0, y2-y1)
    pred_box = (xa2-xa1) * (ya2-ya1)
    correct_box = (xb2-xb1) * (yb2-yb1)

    iou = inter / (pred_box + correct_box - inter)

    return iou

def change_end(correct, width, height):
    x_center = correct[1] * width
    y_center = correct[2] * height
    w = correct[3] * width
    h = correct[4] * height

    # 左上
    x_min = x_center - (w / 2)
    y_min = y_center - (h / 2)

    # 右下
    x_max = x_center + (w / 2)
    y_max = y_center + (h / 2)

    return (x_min, y_min, x_max, y_max)

if __name__ == '__main__':

    start_time = time.time()
    # 事前学習済みのyolov8nモデル
    model = YOLO('runs/train/yolo_training/weights/best.pt')

    test_path = 'datasets_new_back/images/test'
    label_path = 'datasets_new_back/labels/test'
    save_path = 'runs/detect/test_results_back'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)

    image_paths = glob.glob(os.path.join(test_path, '*.'))
    image_paths = sorted(image_paths, key=lambda x: int(''.join(filter(str.isdigit,x))))
    images_num = len(image_paths)
    objects_num = 9
    dis = 0
    accu_total = 0
    accu_pre = 0
    accu_reca = 0

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        file_name = os.path.splitext(image_name)[0]

        # 画像サイズ取得
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        start_time1 = time.time()

        results = model(image_path, conf=0.3, iou=0.5)

        end_time1 = time.time()
        execution_time = end_time1 - start_time1

        predictions = results[0].boxes
        pred_boxes = []
        for p in predictions:
            pred_boxes.append((p.cls.item(), p.xywhn[0][0].item(), p.xywhn[0][1].item(), p.xywhn[0][2].item(), p.xywhn[0][3].item()))

        correct_boxes = correct(os.path.join(label_path, f'{file_name}.txt'))

        # 位置誤差（IoU閾値以上で計算）
        position_errors = []
        total_dis = 0
        for pred_box in pred_boxes:
            min_box = None
            min_distance = float('inf')
            for correct_box in correct_boxes:
                if pred_box[0] == correct_box[0]:
                    cor_bbox = change_end(correct_box, width, height)
                    pred_bbox = change_end(pred_box, width, height)
                    iou = calculate_iou(pred_bbox, cor_bbox)
                    if iou >= 0.5:  # IoUが閾値以上の場合のみ位置誤差を計算
                        distance = np.sqrt((pred_box[1] - correct_box[1])**2 + (pred_box[2] - correct_box[2])**2)
                        if distance < min_distance:
                            min_distance = distance
                            min_box = correct_box
            if min_box is not None:
                position_error = calculate_position_error(pred_box, min_box)
                position_errors.append(position_error)
                total_dis += position_error

        if len(position_errors) == 0:
            dis_average = 0
        else:
            dis_average = total_dis / len(position_errors)
        print(dis_average)
        dis += dis_average

        accu, precision, recall, TP, FP, FN = accuracy(predictions, correct_boxes)
        accu_per, precision_per, recall_per = accu * 100, precision * 100, recall * 100
        accu_total += accu_per
        accu_pre += precision_per
        accu_reca += recall_per
        print(TP, FP, FN)

        img = results[0].plot()
        result_image = os.path.join(save_path, f'{file_name}.jpg')
        cv2.imwrite(result_image, img)

        txt_file = os.path.join(save_path, f'{file_name}.txt')
        results[0].save_txt(txt_file)

    cv2.destroyAllWindows()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("平均位置誤差:", round(dis/images_num, 3))
    print("accuracy:", round(accu_total / images_num, 3))
    print("precision:", round(accu_pre / images_num, 3))
    print("recall:", round(accu_reca / images_num, 3))
    print("画像の数:", images_num)
    print("実行時間:", round(elapsed_time/images_num, 3))
