import cv2
import numpy as np
from collections import OrderedDict
import os
import glob
from scipy.optimize import linear_sum_assignment

# HÀM TÍNH INTERSECTION OVER UNION (IOU)


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# HÀM PHÁT HIỆN BÃO (không đổi)


def detect_rain_cells(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow_orange = np.array([15, 100, 100])
    upper_yellow_orange = np.array([45, 255, 255])
    final_mask = cv2.bitwise_or(cv2.inRange(
        hsv_image, lower_red1, upper_red1), cv2.inRange(hsv_image, lower_red2, upper_red2))
    final_mask = cv2.bitwise_or(final_mask, cv2.inRange(
        hsv_image, lower_yellow_orange, upper_yellow_orange))
    contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:
            bbox = cv2.boundingRect(contour)
            detected_boxes.append(bbox)
    return detected_boxes


# --- CHƯƠNG TRÌNH CHÍNH VỚI KIẾN TRÚC MỚI ---
if __name__ == '__main__':
    main_folder = 'Rad_images'
    output_folder = 'tracking_results_continuous_detection'
    os.makedirs(output_folder, exist_ok=True)
    event_folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

    if not event_folders:
        print(f"Lỗi: Không tìm thấy thư mục con nào trong '{main_folder}'.")

    for folder in event_folders:
        event_name = os.path.basename(folder)
        print(f"\n--- Đang xử lý sự kiện: {event_name} ---")
        image_files = sorted(glob.glob(os.path.join(folder, '*.*')))
        if not image_files:
            print(
                f"Cảnh báo: Không tìm thấy file ảnh nào trong '{folder}'. Bỏ qua.")
            continue

        # Dictionary để quản lý đối tượng (ID, bbox, màu, đường đi, bộ đếm biến mất)
        tracked_objects = OrderedDict()
        nextObjectID = 0
        max_disappeared = 5  # Số khung hình cho phép một tracker biến mất

        sample_image = cv2.imread(image_files[0])
        height, width, _ = sample_image.shape
        video_path = os.path.join(output_folder, f'tracking_{event_name}.mp4')
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (width, height))

        for i, image_path in enumerate(image_files):
            print(
                f"  -> Đang xử lý khung hình {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            # Lấy ra danh sách các tracker và ID đang hoạt động
            active_trackers = cv2.legacy.MultiTracker_create()
            active_objects = OrderedDict()
            for obj_id, data in tracked_objects.items():
                if data['disappeared'] == 0:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    active_trackers.add(tracker, frame, data['bbox'])
                    active_objects[obj_id] = data

            # 1. THEO DÕI: Cập nhật vị trí từ các tracker đang hoạt động
            success, boxes = active_trackers.update(frame)

            # Cập nhật bbox mới cho các đối tượng
            active_ids = list(active_objects.keys())
            for j, box in enumerate(boxes):
                obj_id = active_ids[j]
                tracked_objects[obj_id]['bbox'] = tuple(box)

            # 2. PHÁT HIỆN: Chạy phát hiện trong mỗi khung hình
            detected_boxes = detect_rain_cells(frame)

            # 3. ĐỐI CHIẾU: Ghép cặp tracker và detection
            tracked_bboxes = [data['bbox']
                              for data in tracked_objects.values()]

            # Tính ma trận chi phí IoU (cost = 1 - IoU)
            iou_matrix = np.zeros(
                (len(tracked_bboxes), len(detected_boxes)), dtype="float")
            for t, track_box in enumerate(tracked_bboxes):
                for d, detect_box in enumerate(detected_boxes):
                    iou_matrix[t, d] = calculate_iou(track_box, detect_box)

            # Dùng Hungarian để tối ưu việc ghép cặp
            # `cost = 1 - iou`, thuật toán sẽ tìm cách minimize cost, tức là maximize iou
            rows, cols = linear_sum_assignment(1 - iou_matrix)

            used_tracks = set()
            used_detections = set()

            # A. Xử lý các cặp được ghép nối
            for r, c in zip(rows, cols):
                # Nếu IoU quá thấp, không coi là một cặp
                if iou_matrix[r, c] < 0.1:
                    continue

                obj_id = list(tracked_objects.keys())[r]
                # Cập nhật bbox theo detection cho chính xác
                tracked_objects[obj_id]['bbox'] = detected_boxes[c]
                # Reset bộ đếm biến mất
                tracked_objects[obj_id]['disappeared'] = 0
                used_tracks.add(r)
                used_detections.add(c)

            # B. Xử lý các tracker không được ghép nối (bão đã tan)
            for r in set(range(len(tracked_bboxes))).difference(used_tracks):
                obj_id = list(tracked_objects.keys())[r]
                tracked_objects[obj_id]['disappeared'] += 1

            # C. Xử lý các detection không được ghép nối (bão mới xuất hiện)
            for c in set(range(len(detected_boxes))).difference(used_detections):
                new_id = nextObjectID
                hue = (new_id * 40) % 180
                hsv_color = np.uint8([[[hue, 255, 255]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

                tracked_objects[new_id] = {
                    'bbox': detected_boxes[c], 'path': [],
                    'color': tuple(bgr_color.tolist()), 'disappeared': 0
                }
                nextObjectID += 1

            # VẼ KẾT QUẢ VÀ DỌN DẸP
            final_objects_to_draw = OrderedDict()
            for obj_id, data in tracked_objects.items():
                # Chỉ vẽ các đối tượng chưa biến mất quá lâu
                if data['disappeared'] <= max_disappeared:
                    final_objects_to_draw[obj_id] = data
                    box = data['bbox']
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                    centroid = (int(box[0] + box[2] / 2),
                                int(box[1] + box[3] / 2))
                    data['path'].append(centroid)

                    color = data['color']
                    cv2.rectangle(frame, p1, p2, color, 2, 1)
                    cv2.putText(
                        frame, f"ID {obj_id}", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    path = data['path']
                    for k in range(1, len(path)):
                        cv2.line(frame, path[k - 1], path[k], color, 2)

            # Cập nhật lại danh sách đối tượng sau khi đã dọn dẹp
            tracked_objects = final_objects_to_draw

            video_writer.write(frame)
            # cv2.imshow(f"Tracking Event: {event_name}", frame)

            # if cv2.waitKey(50) & 0xFF == ord('q'):
            #     break

        video_writer.release()
        print(
            f"--- Hoàn thành xử lý sự kiện '{event_name}'. Kết quả đã lưu tại: {video_path} ---")

    cv2.destroyAllWindows()
    print("\nTất cả các sự kiện đã được xử lý.")
