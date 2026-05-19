import numpy as np

def parse_loc(loader, locdata, threshold=0.001):
    """
    Chuyển model output (Cutout) sang tọa độ XY và Merging NMS.
    locdata: (720, 3) -> [objectness, dx, dy]
    """
    # 1. Sigmoid đưa về xác suất
    obj_scores = 1.0 / (1.0 + np.exp(-locdata[:, 0]))
    
    # 2. Tính tọa độ thực (Không cần Polar Grid)
    # Tọa độ dự đoán = Vị trí tia LiDAR + Offset (dx, dy)
    scan_xy = loader.SCAN_POINTS # (720, 2)
    pred_xy = scan_xy + locdata[:, 1:3]

    # 3. Lọc theo ngưỡng
    mask = obj_scores >= threshold
    if not np.any(mask): return None
    
    scores, coords = obj_scores[mask], pred_xy[mask]
    
    # Sắp xếp theo score giảm dần
    order = np.argsort(-scores)
    scores, coords = scores[order], coords[order]

    # 4. Merging NMS (Giữ nguyên logic gộp cụm của bạn)
    merged_dets, merged_counts = [], []
    for i in range(len(scores)):
        cur_s, cur_xy = scores[i], coords[i]
        found = False
        for j in range(len(merged_dets)):
            dist = np.hypot(cur_xy[0] - merged_dets[j][1], cur_xy[1] - merged_dets[j][2])
            if dist <= 1.5 * loader.HARDCODED_PERSON_RADIUS:
                _, ox, oy = merged_dets[j]
                cnt = merged_counts[j]
                alpha = 1.0 / (cnt + 1)
                new_x = alpha * cur_xy[0] + (1 - alpha) * ox
                new_y = alpha * cur_xy[1] + (1 - alpha) * oy
                merged_dets[j] = (max(cur_s, merged_dets[j][0]), new_x, new_y)
                merged_counts[j] += 1
                found = True
                break
        if not found:
            merged_dets.append((cur_s, cur_xy[0], cur_xy[1]))
            merged_counts.append(1)

    return np.array(merged_dets, dtype=np.float32)