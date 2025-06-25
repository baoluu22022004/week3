import cv2
import numpy as np
from collections import OrderedDict
import os
import glob
from scipy.optimize import linear_sum_assignment


# Intersection over Union (IoU), with boxA and boxB in xywh-format

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# HÀM PHÁT HIỆN BÃO (CẬP NHẬT ĐỂ CHỈ TRẢ VỀ TOP 5)


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

    # --- THAY ĐỔI CHÍNH NẰM Ở ĐÂY ---
    detections_with_area = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Tăng ngưỡng diện tích để lọc nhiễu ban đầu tốt hơn
        if area > 300:
            bbox = cv2.boundingRect(contour)
            detections_with_area.append((area, bbox))

    # Sắp xếp các phát hiện theo diện tích giảm dần
    sorted_detections = sorted(
        detections_with_area, key=lambda x: x[0], reverse=True)

    # Chỉ lấy 5 cơn bão lớn nhất
    top_5_detections = sorted_detections[:5]

    # Trích xuất bounding box từ top 5
    top_5_boxes = [item[1] for item in top_5_detections]

    return top_5_boxes
    # --- KẾT THÚC THAY ĐỔI ---


# --- CHƯƠNG TRÌNH CHÍNH (ĐÃ CẬP NHẬT) ---
if __name__ == '__main__':
    main_folder = 'Rad_images'
    output_folder = 'tracking_results_top5'  # Folder kết quả mới
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

        tracked_objects = OrderedDict()
        nextObjectID = 0
        max_disappeared = 5

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

            # --- Logic theo dõi và đối chiếu (không đổi, tự động xử lý top 5) ---
            active_trackers = cv2.legacy.MultiTracker_create()
            for obj_id, data in tracked_objects.items():
                if data['disappeared'] == 0:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    active_trackers.add(tracker, frame, data['bbox'])
            success, boxes = active_trackers.update(frame)
            active_ids = [
                obj_id for obj_id, data in tracked_objects.items() if data['disappeared'] == 0]
            for j, box in enumerate(boxes):
                obj_id = active_ids[j]
                tracked_objects[obj_id]['bbox'] = tuple(box)

            # Hàm detect_rain_cells giờ chỉ trả về top 5
            detected_boxes = detect_rain_cells(frame)

            tracked_bboxes_ids = list(tracked_objects.keys())
            if len(tracked_bboxes_ids) > 0 and len(detected_boxes) > 0:
                iou_matrix = np.zeros(
                    (len(tracked_bboxes_ids), len(detected_boxes)), dtype="float")
                for t, obj_id in enumerate(tracked_bboxes_ids):
                    for d, detect_box in enumerate(detected_boxes):
                        iou_matrix[t, d] = calculate_iou(
                            tracked_objects[obj_id]['bbox'], detect_box)
                rows, cols = linear_sum_assignment(1 - iou_matrix)
                used_tracks = set()
                used_detections = set()
                for r, c in zip(rows, cols):
                    if iou_matrix[r, c] < 0.1:
                        continue
                    obj_id = tracked_bboxes_ids[r]
                    tracked_objects[obj_id]['bbox'] = detected_boxes[c]
                    tracked_objects[obj_id]['disappeared'] = 0
                    used_tracks.add(r)
                    used_detections.add(c)
                for r in set(range(len(tracked_bboxes_ids))).difference(used_tracks):
                    obj_id = tracked_bboxes_ids[r]
                    tracked_objects[obj_id]['disappeared'] += 1
                for c in set(range(len(detected_boxes))).difference(used_detections):
                    new_id = nextObjectID
                    hue = (new_id * 40) % 180
                    hsv_color = np.uint8([[[hue, 255, 255]]])
                    bgr_color = cv2.cvtColor(
                        hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                    tracked_objects[new_id] = {'bbox': detected_boxes[c], 'path': [], 'color': tuple(
                        bgr_color.tolist()), 'disappeared': 0, 'velocity': np.array([0, 0])}
                    nextObjectID += 1
            elif len(detected_boxes) > 0:
                for c in range(len(detected_boxes)):
                    new_id = nextObjectID
                    hue = (new_id * 40) % 180
                    hsv_color = np.uint8([[[hue, 255, 255]]])
                    bgr_color = cv2.cvtColor(
                        hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                    tracked_objects[new_id] = {'bbox': detected_boxes[c], 'path': [], 'color': tuple(
                        bgr_color.tolist()), 'disappeared': 0, 'velocity': np.array([0, 0])}
                    nextObjectID += 1

            # --- VẼ KẾT QUẢ ---
            final_objects_to_draw = OrderedDict()
            for obj_id, data in tracked_objects.items():
                if data['disappeared'] <= max_disappeared:
                    final_objects_to_draw[obj_id] = data
                    box = data['bbox']
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                    centroid = (int(box[0] + box[2] / 2),
                                int(box[1] + box[3] / 2))
                    previous_centroid = data['path'][-1] if len(
                        data['path']) > 0 else centroid
                    data['velocity'] = np.array(
                        centroid) - np.array(previous_centroid)
                    data['path'].append(centroid)
                    color = data['color']
                    cv2.rectangle(frame, p1, p2, color, 2, 1)
                    cv2.putText(
                        frame, f"ID {obj_id}", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    velocity = data['velocity']
                    end_point = (
                        int(centroid[0] + velocity[0] * 3), int(centroid[1] + velocity[1] * 3))
                    cv2.arrowedLine(frame, centroid, end_point,
                                    (0, 0, 255), 2, tipLength=0.4)

            # --- COMMENT LẠI PHẦN TÍNH TOÁN XU HƯỚNG CHUNG ---
            # (Vì giờ đây việc quan sát 5 mũi tên đã đủ trực quan)
            # active_velocities = [data['velocity'] for data in final_objects_to_draw.values()]
            # if active_velocities:
            #     median_vector = np.median(np.array(active_velocities), axis=0)
            #     overall_speed = np.linalg.norm(median_vector)
            #     overall_angle = np.degrees(np.arctan2(median_vector[1], median_vector[0]))
            #     trend_text = f"Overall Trend: {overall_speed:.1f} px/f @ {overall_angle:.0f} deg"
            #     cv2.putText(frame, trend_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
