import numpy as np

def parse_loc(loader, locdata, threshold=0.001):
    """
    Chuyển model output sang tọa độ XY và Merging NMS.
    """
    # 1. Sigmoid đưa về xác suất
    a_obj = 1.0 / (1.0 + np.exp(-locdata[:, :, 0]))
    
    # 2. Hoàn tác Scale và chuyển Arc -> Angle
    a_reg = locdata[:, :, 1:3] / loader.ANCHOR_REGRESSION_SCALE
    a_reg[:, :, 1] /= (loader.grid_polar[:, :, 0] + 1e-8)

    # 3. Tính tọa độ thực
    a_regr_polar = loader.grid_polar + a_reg
    r, theta = a_regr_polar[:, :, 0], a_regr_polar[:, :, 1]
    a_regr_xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=2)

    # 4. Lọc và Merging NMS
    a_obj_f, a_xy_f = a_obj.reshape(-1), a_regr_xy.reshape(-1, 2)
    mask = a_obj_f >= threshold
    if not np.any(mask): return None
    
    scores, coords = a_obj_f[mask], a_xy_f[mask]
    order = np.argsort(-scores)
    scores, coords = scores[order], coords[order]

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