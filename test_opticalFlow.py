import cv2
import numpy as np
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import os
import glob

# LỚP TRACKER TỐI ƯU CUỐI CÙNG (TÍCH HỢP OPTICAL FLOW)


class AdvancedTracker:
    def __init__(self, maxDisappeared=15, distance_threshold=25,
                 weight_dist=0.6, weight_size=0.2, weight_shape=0.2):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.maxDisappeared = maxDisappeared

        self.distance_threshold = distance_threshold
        self.W_DIST = weight_dist
        self.W_SIZE = weight_size
        self.W_SHAPE = weight_shape

    def register(self, detection):
        # ... (logic không thay đổi)
        obj_id = self.nextObjectID
        hue = (obj_id * 40) % 180
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        color_tuple = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
        self.objects[obj_id] = {'centroid': detection['centroid'], 'area': detection['area'], 'contour': detection['contour'], 'path': [
            detection['centroid']], 'velocity': np.array([0, 0]), 'disappeared_count': 0, 'color': color_tuple}
        self.nextObjectID += 1

    def deregister(self, objectID):
        # ... (logic không thay đổi)
        del self.objects[objectID]

    # *** PHƯƠNG THỨC UPDATE ĐƯỢC NÂNG CẤP ***
    def update(self, detections, current_frame, previous_frame):
        if len(detections) == 0:
            for objectID in list(self.objects.keys()):
                self.objects[objectID]['disappeared_count'] += 1
                if self.objects[objectID]['disappeared_count'] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(detections[i])
            return self.objects

        # *** BƯỚC 1: TÍNH TOÁN LUỒNG QUANG HỌC GIỮA 2 KHUNG HÌNH (CHỈ 1 LẦN) ***
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # `flow` là một ma trận chứa vector (dx, dy) cho mỗi pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # *** BƯỚC 2: GHÉP CẶP DỰA TRÊN ĐA ĐẶC TÍNH (LOGIC CŨ VẪN GIỮ NGUYÊN) ***
        objectIDs = list(self.objects.keys())
        num_objects = len(objectIDs)
        num_detections = len(detections)
        cost_matrix = np.zeros((num_objects, num_detections))

        for i in range(num_objects):
            # ... (logic tính cost matrix đa đặc tính không thay đổi)
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

        # *** BƯỚC 3: CẬP NHẬT VẬN TỐC BẰNG OPTICAL FLOW CHO CÁC CẶP ĐÃ GHÉP ***
        for row, col in zip(rows, cols):
            if cost_matrix[row, col] >= 10000:
                continue

            objectID = objectIDs[row]
            obj = self.objects[objectID]
            det = detections[col]

            # --- PHẦN THAY THẾ QUAN TRỌNG ---
            # Tạo mặt nạ cho cơn bão mới được phát hiện
            mask = np.zeros_like(curr_gray)
            cv2.drawContours(mask, [det['contour']], -1, 255, -1)

            # Lấy các vector luồng quang học chỉ trong khu vực cơn bão
            storm_flow_vectors = flow[mask == 255]

            if len(storm_flow_vectors) > 0:
                # Tính vector trung vị để loại bỏ nhiễu và có kết quả ổn định
                median_flow = np.median(storm_flow_vectors, axis=0)
                # Gán vector vận tốc mới, chính xác hơn
                obj['velocity'] = median_flow
            else:
                # Nếu không có flow, dùng lại cách cũ như một phương án dự phòng
                obj['velocity'] = det['centroid'] - obj['centroid']
            # --- KẾT THÚC PHẦN THAY THẾ ---

            # Cập nhật các thông tin khác như cũ
            obj['centroid'] = det['centroid']
            obj['area'] = det['area']
            obj['contour'] = det['contour']
            obj['path'].append(det['centroid'])
            obj['disappeared_count'] = 0

            usedRows.add(row)
            usedCols.add(col)

        # ... (logic xử lý unmatched và register/deregister không đổi)
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

# --- CÁC HÀM CÒN LẠI KHÔNG THAY ĐỔI ---


def detect_rain_cells(image):
    # ... (code không đổi)
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
            centroid = np.array(
                [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            detections.append(
                {'centroid': centroid, 'area': area, 'contour': contour})
    return detections


def draw_results(frame, objects):
    # ... (code không đổi)
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


# --- CHƯƠNG TRÌNH CHÍNH ĐƯỢC CẬP NHẬT ĐỂ XỬ LÝ `previous_frame` ---
if __name__ == '__main__':
    main_folder = 'Rad_images'
    output_folder = 'tracking_results_optical_flow'  # Folder kết quả mới
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

        tracker = AdvancedTracker()  # Có thể thêm các trọng số tùy chỉnh ở đây

        # Biến để lưu trữ khung hình trước đó
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

            # Chỉ bắt đầu xử lý từ khung hình thứ hai trở đi
            if previous_frame is not None:
                detections = detect_rain_cells(current_frame)

                # Gọi hàm update với cả 3 tham số
                objects = tracker.update(
                    detections, current_frame, previous_frame)

                output_frame = draw_results(current_frame, objects)
                video_writer.write(output_frame)
                # cv2.imshow(f"Tracking Event: {event_name}", output_frame)
                # if cv2.waitKey(50) & 0xFF == ord('q'):
                #     break

            # Cập nhật khung hình trước đó cho vòng lặp tiếp theo
            previous_frame = current_frame.copy()

        video_writer.release()
        print(
            f"--- Hoàn thành xử lý sự kiện '{event_name}'. Kết quả đã lưu tại: {video_path} ---")

    cv2.destroyAllWindows()
    print("\nTất cả các sự kiện đã được xử lý.")
