from dataclasses import dataclass
import cv2
import numpy as np
import sys


def get_mask(img):
    hsv = cv2.cvtColor(
        cv2.medianBlur(img, min(img.shape[:2])//2048*2+1),
        cv2.COLOR_BGR2HSV
    )
    return cv2.inRange(
        hsv,
        (180*1.5/5, 256//4, 256//20),
        (180*3/5, 256, 256*7//8)
    )


def get_edge_contour(mask_blackboard):
    contours, _ = cv2.findContours(
        mask_blackboard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [
        c_ for c in contours
        if (c_ :=
            c[
                cv2.inRange(
                    c,
                    (3, 3),
                    (mask_blackboard.shape[1] - 4, mask_blackboard.shape[0]-4)
                )[:, 0].astype(np.bool8)
            ]).size
    ]
    return max(map(lambda x: (cv2.contourArea(x), x),
                   contours), key=lambda x: x[0])


@ dataclass
class LinesInfo:
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_length: int
    max_length: int
    area: int
    mask_blackboard: np.array


def get_lines_step1(area, contour, mask_blackboard):
    min_x = contour[:, 0, 0].min()
    max_x = contour[:, 0, 0].max()
    min_y = contour[:, 0, 1].min()
    max_y = contour[:, 0, 1].max()
    min_length = min(max_x - min_x, max_y - min_y)
    max_length = max(max_x - min_x, max_y - min_y)
    lines = cv2.HoughLinesPointSet(contour - (min_x, min_y),
                                   lines_max=20,
                                   threshold=max_length//20,
                                   min_rho=0,
                                   max_rho=max_length+1,
                                   rho_step=min_length/400,
                                   min_theta=-np.pi,
                                   max_theta=np.pi,
                                   theta_step=np.pi*0.5/180
                                   )
    return lines, LinesInfo(min_x, max_x, min_y, max_y, min_length, max_length, area, mask_blackboard)


def get_lines_step2(lines, info: LinesInfo):
    lines = lines[lines[:, 0, 2].argsort()]
    lines_choice = [True]*len(lines)
    old_index = 0
    for i in range(1, len(lines)):
        if lines[i, 0, 2]-lines[i-1, 0, 2] <= np.pi/18/2 and np.abs(lines[i, 0, 1]-lines[i-1, 0, 1]) < info.max_length//64:
            if lines[i, 0, 0] > lines[old_index, 0, 0]:
                lines_choice[old_index] = False
                old_index = i
                lines_choice[i] = True
            else:
                lines_choice[i] = False
        else:
            old_index = i
    return lines[:, :, 1:][lines_choice], info


def get_lines_step3(lines, info: LinesInfo):
    assert len(lines) >= 4
    min_x = info.min_x
    max_x = info.max_x
    min_y = info.min_y
    max_y = info.max_y
    max_length = info.max_length
    area = info.area

    def intersection(line1, line2):
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        if -np.pi/6 < theta1 - theta2 < np.pi/6:
            return None
        A = np.array([
            (np.cos(theta1), np.sin(theta1)),
            (np.cos(theta2), np.sin(theta2))
        ])
        b = np.array(((rho1,), (rho2,)))
        try:
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            if max(x0-max_x, y0-max_y, min_x-x0, min_y-y0) < max_length/2:
                return np.array(((min_x+x0, min_y+y0),))
        except np.linalg.LinAlgError:
            return None

    def direction(p1, p2, p3):
        return np.linalg.det(np.concatenate((p1-p2, p3-p2)))

    def is_convex(points):
        sign = np.sign(a1 := direction(points[-2], points[-1], points[0]))
        if np.sign(direction(points[-1], points[0], points[1])) != sign:
            return False
        if np.sign(a2 := direction(points[0], points[1], points[2])) != sign:
            return False
        if np.sign(direction(points[1], points[2], points[3])) != sign:
            return False
        return (a1+a2)//2 if (a1+a2)//2 < area * 2 else False

    board_lines = []
    points = tuple(tuple(intersection(lines[i], lines[j])
                         for j in range(i)) for i in range(len(lines)))
    for i in range(len(lines)-2):
        for j in range(i+1, len(lines)):
            if points[j][i] is None:
                continue
            for k in range(j+1, len(lines)):
                if points[k][i] is None:
                    continue
                for l in range(i+1, len(lines)):
                    if l == k or l == j:
                        continue
                    if (points[k][l] if k > l else points[l][k]) is None or \
                            (points[l][j] if l > j else points[j][l]) is None:
                        continue
                    nodes = np.array((points[j][i], points[k][i],
                                      points[k][l] if k > l else points[l][k],
                                      points[l][j] if l > j else points[j][l]))
                    if not is_convex(nodes):
                        continue
                    max_x_estimated = max(max(nodes[:, 0, 0]), max_x)
                    max_y_estimated = max(max(nodes[:, 0, 1]), max_y)
                    min_x_estimated = min(min(nodes[:, 0, 0]), min_x)
                    min_y_estimated = min(min(nodes[:, 0, 1]), min_y)
                    y_len_estimated = max_y_estimated-min_y_estimated+1
                    x_len_estimated = max_x_estimated-min_x_estimated+1
                    shape_estimated = cv2.fillConvexPoly(
                        np.zeros(
                            (y_len_estimated, x_len_estimated),
                            dtype=np.uint8
                        ),
                        nodes - (min_x_estimated, min_y_estimated),
                        255
                    )

                    shape_actual = np.zeros(
                        (y_len_estimated, x_len_estimated),
                        dtype=np.uint8
                    )
                    shape_actual[min_y-min_y_estimated:max_y-min_y_estimated+1,
                                 min_x-min_x_estimated:max_x - min_x_estimated+1] \
                        = info.mask_blackboard[min_y:max_y+1, min_x:max_x+1]

                    not_coverd_area = np.count_nonzero(
                        np.logical_and(
                            np.logical_not(shape_estimated),
                            shape_actual
                        )
                    )
                    diff_area = np.count_nonzero(
                        np.logical_xor(shape_estimated, shape_actual)
                    )
                    if (not_coverd_area < area//10):
                        board_lines.append((nodes, not_coverd_area, diff_area))
    if len(board_lines) == 0:
        raise ValueError("Cannot find a convex quadrangle")
    return min(board_lines, key=lambda x: x[2]+x[1])[0]


def adjust(img, points: np.array):
    board_h = 2048
    board_w = 4096
    pts1 = np.float32(points)
    # pylint: ignore disable=unsupported-assignment-operation,unsubscriptable-object
    pts1 = pts1[pts1[:, 0, 0].argsort()]
    # pylint: ignore disable=unsupported-assignment-operation,unsubscriptable-object
    pts1[0:2] = pts1[pts1[0:2, 0, 1].argsort()]
    # pylint: ignore disable=unsupported-assignment-operation,unsubscriptable-object
    pts1[2:4] = pts1[2:4][pts1[2:4, 0, 1].argsort()[::-1]]
    pts2 = np.array(((0, 0), (0, board_h), (board_w, board_h),
                    (board_w, 0)), np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (board_w, board_h))
    black_count = np.count_nonzero(blacks := (dst == 0).all(axis=2))
    total_count = dst.shape[0] * dst.shape[1]
    means = np.array(np.ndarray.mean(dst, axis=(0, 1)) *
                     (1-black_count/total_count), np.uint8)
    dst[blacks] = means
    return dst


def segment_strokes(dst):
    def circ_kernel(r): return (
        np.add(*map(lambda x: x**2, np.ogrid[-r:r+1, -r:r+1])) <= r**2).astype(np.uint8)
    dst_new = cv2.erode(dst, circ_kernel(1))
    backcolor = cv2.erode(cv2.dilate(dst_new, circ_kernel(5)), circ_kernel(8))
    forecolor = dst_new
    return (forecolor > backcolor + np.uint8(5)).all(axis=2).astype(np.uint8) * np.uint8(255)


def strokes_whitebalance(dst, mask):
    dst_stroke = dst.copy()  # - backcolor
    dst_stroke[np.where(mask == 0)] = (0, 0, 0)
    mean = dst_stroke[100:-100, 200:-
                      200][np.where(mask[100:-100, 200:-200] != 0)].mean(axis=0)
    k = max(mean.mean(), 128+64) / mean
    overflew_mask = (dst_stroke > 255 // k).any(axis=2)
    overflew_axis = np.where(overflew_mask)
    not_over_axis = np.where(np.logical_not(overflew_mask))
    dst_stroke[not_over_axis] = \
        (dst_stroke[not_over_axis] * k).astype(np.uint8)
    k_max = (dst_stroke[overflew_axis] / mean.astype(np.float)).max(axis=1)
    k_array = np.float(255) / (mean * np.array((k_max, k_max, k_max), np.float).T)
    dst_stroke[overflew_axis] = (
        dst_stroke[overflew_axis] * k_array).astype(np.uint8)
    return dst_stroke


def draw_line(img, line, info: LinesInfo, color=(255,255,255)):
    rho, theta = line[0, -2:]  # 获取极值ρ长度和θ角度
    a = np.cos(theta)  # 获取角度cos值
    b = np.sin(theta)  # 获取角度sin值
    x0 = info.min_x + a * rho  # 获取x轴值
    y0 = info.min_y + b * rho  # 获取y轴值　　x0和y0是直线的中点
    max_length = img.shape[0] + img.shape[1]
    x1 = int(x0 - max_length * (-b))
    y1 = int(y0 - max_length * (a))  # 获取这条直线最大值点y1
    x2 = int(x0 + max_length * (-b))  # 获取这条直线最小值点x2　　
    y2 = int(y0 + max_length * (a))  # 获取这条直线最小值点y2　　其中*1000是内部规则
    cv2.line(img, (x1, y1), (x2, y2), color,
             max(img.shape[:2])//512)  # 开始划线


def main():
    input_filename = sys.argv[1]
    img = cv2.imread(input_filename, cv2.IMREAD_COLOR)
    mask_blackboard = get_mask(img)
    area, c = get_edge_contour(mask_blackboard)
    points = get_lines_step3(
        *get_lines_step2(*get_lines_step1(area, c, mask_blackboard)))
    dst_adjusted = adjust(img, points)
    cv2.imwrite("adjusted.jpg", dst_adjusted)
    dst_bw = segment_strokes(dst_adjusted)
    cv2.imwrite("bw.jpg", dst_bw)
    dst = strokes_whitebalance(dst_adjusted, dst_bw)
    cv2.imwrite("dst.jpg", dst)


if __name__ == "__main__":
    main()
