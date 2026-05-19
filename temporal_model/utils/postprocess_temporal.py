import numpy as np

def canonical_to_global(scan_r, scan_phi, dx, dy):
    """ Chuyển đổi từ tọa độ địa phương (Canonical) về tọa độ toàn cục (Polar) """
    tmp_y = scan_r + dy
    tmp_phi = np.arctan2(dx, tmp_y)
    dets_phi = tmp_phi + scan_phi
    dets_r = tmp_y / np.cos(tmp_phi)
    return dets_r, dets_phi

def parse_loc(loader, locdata, raw_scan, threshold=0.001):
    """
    Chuyển model output (Cutout) sang tọa độ XY thực và Merging NMS.
    - locdata: (720, 3) -> [objectness, dx_local, dy_local]
    - raw_scan: (720,) -> Phải truyền bản đồ khoảng cách của frame hiện tại vào đây!
    """
    # 1. Sigmoid đưa về xác suất
    obj_scores = 1.0 / (1.0 + np.exp(-locdata[:, 0]))
    
    # Lọc threshold sớm để tăng tốc độ
    mask = obj_scores >= threshold
    if not np.any(mask): return None
    
    scores = obj_scores[mask]
    
    # 2. Hoàn tác Scale (Chia cho 10.0 như đã cấu hình trong Dataloader)
    dx_local = locdata[mask, 1] / 10.0
    dy_local = locdata[mask, 2] / 10.0
    
    # 3. Phục hồi tọa độ Polar Toàn cục (Global) sử dụng raw_scan
    scan_r = raw_scan[mask]
    scan_phi = loader.SCAN_ANGLES[mask]
    
    dets_r, dets_phi = canonical_to_global(scan_r, scan_phi, dx_local, dy_local)
    
    # Chuyển từ hệ tọa độ Cực (r, phi) sang hệ tọa độ Descartes (X, Y)
    pred_x = dets_r * np.cos(dets_phi)
    pred_y = dets_r * np.sin(dets_phi)
    coords = np.stack([pred_x, pred_y], axis=1)

    # 4. Sắp xếp theo score giảm dần
    order = np.argsort(-scores)
    scores, coords = scores[order], coords[order]

    # 5. Merging NMS (Giữ nguyên logic gộp cụm)
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