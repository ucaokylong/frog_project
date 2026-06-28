import numpy as np

def parse_loc(loader, locdata, threshold=0.01):
    """
    Chuyển model output sang tọa độ XY và Merging bằng tư tưởng Weighted Boxes Fusion (WBF).
    Tọa độ được nội suy tỷ lệ thuận với confidence score để chống lệch tâm.
    """
    # 1. Sigmoid đưa về xác suất
    a_obj = 1.0 / (1.0 + np.exp(-locdata[:, :, 0]))
    
    # 2. Hoàn tác Scale và chuyển Arc -> Angle
    a_reg = locdata[:, :, 1:3] / loader.ANCHOR_REGRESSION_SCALE
    a_reg[:, :, 1] /= (loader.grid_polar[:, :, 0] + 1e-8)

    # 3. Tính tọa độ thực (Hệ tọa độ cos/sin đã chuẩn)
    a_regr_polar = loader.grid_polar + a_reg
    r, theta = a_regr_polar[:, :, 0], a_regr_polar[:, :, 1]
    a_regr_xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=2)

    # 4. Lọc bỏ background
    a_obj_f, a_xy_f = a_obj.reshape(-1), a_regr_xy.reshape(-1, 2)
    mask = a_obj_f >= threshold
    if not np.any(mask): return None
    
    # 5. Sắp xếp theo score giảm dần
    scores, coords = a_obj_f[mask], a_xy_f[mask]
    order = np.argsort(-scores)
    scores, coords = scores[order], coords[order]

    # --- KHỞI TẠO WEIGHTED FUSION ---
    # Cấu trúc lưu trữ: [max_score, weighted_x, weighted_y, sum_of_scores]
    merged_clusters = [] 
    
    # Bán kính gom cụm (có thể tinh chỉnh nếu muốn gom gắt hơn)
    merge_distance = 1.5 * loader.HARDCODED_PERSON_RADIUS

    for i in range(len(scores)):
        cur_s, cur_xy = scores[i], coords[i]
        found = False
        
        for j in range(len(merged_clusters)):
            m_s, m_x, m_y, m_sum_s = merged_clusters[j]
            
            # Tính khoảng cách từ điểm hiện tại tới tâm WBF của cụm
            dist = np.hypot(cur_xy[0] - m_x, cur_xy[1] - m_y)
            
            if dist <= merge_distance:
                # Cập nhật tổng trọng số (sum of scores)
                new_sum_s = m_sum_s + cur_s
                
                # Tọa độ mới là trung bình có trọng số (Weighted Average)
                new_x = (m_x * m_sum_s + cur_xy[0] * cur_s) / new_sum_s
                new_y = (m_y * m_sum_s + cur_xy[1] * cur_s) / new_sum_s
                
                # Giữ lại max score làm đại diện cho cụm
                new_s = max(cur_s, m_s)
                
                # Lưu ngược lại vào cụm
                merged_clusters[j] = [new_s, new_x, new_y, new_sum_s]
                found = True
                break
                
        if not found:
            # Nếu không thuộc cụm nào, tạo cụm mới
            merged_clusters.append([cur_s, cur_xy[0], cur_xy[1], cur_s])

    # Trích xuất ra format chuẩn (score, x, y)
    final_dets = [(c[0], c[1], c[2]) for c in merged_clusters]
    return np.array(final_dets, dtype=np.float32)



# import numpy as np

# def parse_loc(loader, locdata, threshold=0.01):
#     """
#     Chuyển model output sang tọa độ XY và Merging NMS.
#     """
#     # 1. Sigmoid đưa về xác suất
#     a_obj = 1.0 / (1.0 + np.exp(-locdata[:, :, 0]))
    
#     # 2. Hoàn tác Scale và chuyển Arc -> Angle
#     a_reg = locdata[:, :, 1:3] / loader.ANCHOR_REGRESSION_SCALE
#     a_reg[:, :, 1] /= (loader.grid_polar[:, :, 0] + 1e-8)

#     # 3. Tính tọa độ thực (Hệ tọa độ cos/sin đã chuẩn)
#     a_regr_polar = loader.grid_polar + a_reg
#     r, theta = a_regr_polar[:, :, 0], a_regr_polar[:, :, 1]
#     a_regr_xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=2)

#     # 4. Lọc và Merging NMS
#     a_obj_f, a_xy_f = a_obj.reshape(-1), a_regr_xy.reshape(-1, 2)
#     mask = a_obj_f >= threshold
#     if not np.any(mask): return None
    
#     scores, coords = a_obj_f[mask], a_xy_f[mask]
#     order = np.argsort(-scores)
#     scores, coords = scores[order], coords[order]

#     merged_dets, merged_counts = [], []
#     for i in range(len(scores)):
#         cur_s, cur_xy = scores[i], coords[i]
#         found = False
#         for j in range(len(merged_dets)):
#             dist = np.hypot(cur_xy[0] - merged_dets[j][1], cur_xy[1] - merged_dets[j][2])
#             if dist <= 1.5 * loader.HARDCODED_PERSON_RADIUS:
#                 _, ox, oy = merged_dets[j]
#                 cnt = merged_counts[j]
#                 alpha = 1.0 / (cnt + 1)
#                 new_x = alpha * cur_xy[0] + (1 - alpha) * ox
#                 new_y = alpha * cur_xy[1] + (1 - alpha) * oy
#                 merged_dets[j] = (max(cur_s, merged_dets[j][0]), new_x, new_y)
#                 merged_counts[j] += 1
#                 found = True
#                 break
#         if not found:
#             merged_dets.append((cur_s, cur_xy[0], cur_xy[1]))
#             merged_counts.append(1)

#     return np.array(merged_dets, dtype=np.float32)