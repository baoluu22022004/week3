import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import os
import glob

# LỚP TRACKER TÍCH HỢP TƯƠNG QUAN MẪU (ĐÃ SỬA LỖI)


class AdvancedTracker:
    def __init__(self, maxDisappeared=15, distance_threshold=75,
                 weight_dist=0.6, weight_size=0.2, weight_shape=0.2):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.maxDisappeared = maxDisappeared

        self.distance_threshold = distance_threshold
        self.W_DIST = weight_dist
        self.W_SIZE = weight_size
        self.W_SHAPE = weight_shape

    def register(self, detection):
        obj_id = self.nextObjectID
        hue = (obj_id * 40) % 180
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        color_tuple = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

        self.objects[obj_id] = {
            'centroid': detection['centroid'], 'area': detection['area'],
            'contour': detection['contour'], 'bbox': detection['bbox'],
            'path': [detection['centroid']], 'velocity': np.array([0, 0]),
            'disappeared_count': 0, 'color': color_tuple
        }
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]

    def update(self, detections, current_frame, previous_frame):
        # *** KHỐI MÃ ĐÃ SỬA LỖI UnboundLocalError ***
        if len(detections) == 0:
            # Lặp qua tất cả các ID đang được theo dõi
            for objectID in list(self.objects.keys()):
                # Tăng bộ đếm biến mất
                self.objects[objectID]['disappeared_count'] += 1

                # Nếu biến mất quá lâu, hủy đăng ký (câu lệnh if đã được đưa vào trong)
                if self.objects[objectID]['disappeared_count'] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        # *** KẾT THÚC SỬA LỖI ***

        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(detections[i])
            return self.objects

        objectIDs = list(self.objects.keys())
        num_objects = len(objectIDs)
        num_detections = len(detections)
        cost_matrix = np.zeros((num_objects, num_detections))

        for i in range(num_objects):
            obj_id = objectIDs[i]
            obj_data = self.objects[obj_id]
            predicted_centroid = obj_data['centroid'] + obj_data['velocity']
            for j in range(num_detections):
                det_data = detections[j]
                dist_cost = dist.euclidean(
                    predicted_centroid, det_data['centroid'])
                if dist_cost > self.distance_threshold:
                    cost_matrix[i, j] = 10000
                    continue
                size_cost = abs(
                    obj_data['area'] - det_data['area']) / max(obj_data['area'], det_data['area'])
                shape_cost = cv2.matchShapes(
                    obj_data['contour'], det_data['contour'], cv2.CONTOURS_MATCH_I1, 0.0)
                total_cost = (self.W_DIST * dist_cost) + \
                    (self.W_SIZE * size_cost) + (self.W_SHAPE * shape_cost)
                cost_matrix[i, j] = total_cost

        rows, cols = linear_sum_assignment(cost_matrix)
        usedRows, usedCols = set(), set()

        for row, col in zip(rows, cols):
            if cost_matrix[row, col] >= 10000:
                continue

            objectID = objectIDs[row]
            obj = self.objects[objectID]
            det = detections[col]
            old_bbox = obj['bbox']
            x, y, w, h = old_bbox
            template = previous_frame[y:y+h, x:x+w]

            if template.shape[0] > 0 and template.shape[1] > 0:
                result = cv2.matchTemplate(
                    current_frame, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                motion_vector = np.array([max_loc[0] - x, max_loc[1] - y])
                obj['velocity'] = motion_vector
            else:
                obj['velocity'] = det['centroid'] - obj['centroid']

            obj['centroid'] = det['centroid']
            obj['area'] = det['area']
            obj['contour'] = det['contour']
            obj['bbox'] = det['bbox']
            obj['path'].append(det['centroid'])
            obj['disappeared_count'] = 0

            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(num_objects)).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.objects[objectID]['disappeared_count'] += 1
            if self.objects[objectID]['disappeared_count'] > self.maxDisappeared:
                self.deregister(objectID)
        unusedCols = set(range(num_detections)).difference(usedCols)
        for col in unusedCols:
            self.register(detections[col])

        return self.objects

# HÀM PHÁT HIỆN VÙNG MƯA (ĐÃ SỬA LỖI ZeroDivisionError)


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

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 150:
            M = cv2.moments(contour)
            # *** THÊM KIỂM TRA ĐỂ TRÁNH LỖI CHIA CHO 0 ***
            if M["m00"] != 0:
                centroid = np.array(
                    [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
                bbox = cv2.boundingRect(contour)
                detections.append(
                    {'centroid': centroid, 'area': area, 'contour': contour, 'bbox': bbox})
            # *** KẾT THÚC SỬA LỖI ***
    return detections


def draw_results(frame, objects):
    for (objectID, data) in objects.items():
        color = data['color']
        centroid = tuple(data['centroid'].astype(int))
        text = f"ID {objectID}"
        cv2.putText(
            frame, text, (centroid[0] - 15, centroid[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, centroid, 5, color, -1)
        path = data['path']
        for i in range(1, len(path)):
            start_point = tuple(path[i - 1].astype(int))
            end_point = tuple(path[i].astype(int))
            cv2.line(frame, start_point, end_point, color, 2)
    return frame


# CHƯƠNG TRÌNH CHÍNH (không đổi)
if __name__ == '__main__':
    main_folder = 'Rad_images'
    output_folder = 'tracking_results_correlation'
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

        tracker = AdvancedTracker()
        previous_frame = None

        sample_image = cv2.imread(image_files[0])
        height, width, _ = sample_image.shape
        video_path = os.path.join(output_folder, f'tracking_{event_name}.mp4')
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (width, height))

        for i, image_path in enumerate(image_files):
            print(
                f"  -> Đang xử lý khung hình {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            current_frame = cv2.imread(image_path)
            if current_frame is None:
                continue

            if previous_frame is not None:
                detections = detect_rain_cells(current_frame)
                objects = tracker.update(
                    detections, current_frame, previous_frame)
                output_frame = draw_results(current_frame, objects)
                video_writer.write(output_frame)
                # cv2.imshow(f"Tracking Event: {event_name}", output_frame)
                # if cv2.waitKey(50) & 0xFF == ord('q'):
                #     break

            previous_frame = current_frame.copy()

        video_writer.release()
        print(
            f"--- Hoàn thành xử lý sự kiện '{event_name}'. Kết quả đã lưu tại: {video_path} ---")

    cv2.destroyAllWindows()
    print("\nTất cả các sự kiện đã được xử lý.")
