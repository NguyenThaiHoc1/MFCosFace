import math
from configparser import ConfigParser

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageColor
from dotmap import DotMap
from numpy.linalg import eig, inv


def rect_to_bb(rect):
    xmin, ymin, xmax, ymax = rect  # top, left, bottom, right

    x1 = ymin  # left
    x2 = ymax  # right
    y1 = xmin  # top
    y2 = xmax  # bottom

    return (x1, x2, y2, x1)


# function
def change_saturation(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s = value * s
    s[s > 255] = 255
    s = np.asarray(s, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def get_avg_saturation(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def change_brightness(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v = value * v
    v[v > 255] = 255
    v = np.asarray(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def get_avg_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def color_the_mask(mask_image, color, intensity):
    assert 0 <= intensity <= 1, "intensity should be between 0 and 1"
    RGB_color = ImageColor.getcolor(color, "RGB")
    RGB_color = (RGB_color[2], RGB_color[1], RGB_color[0])
    orig_shape = mask_image.shape
    bit_mask = mask_image[:, :, 3]
    mask_image = mask_image[:, :, 0:3]

    color_image = np.full(mask_image.shape, RGB_color, np.uint8)
    mask_color = cv2.addWeighted(mask_image, 1 - intensity, color_image, intensity, 0)
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=bit_mask)
    colored_mask = np.zeros(orig_shape, dtype=np.uint8)
    colored_mask[:, :, 0:3] = mask_color
    colored_mask[:, :, 3] = bit_mask
    return colored_mask


def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2;
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


def ellipse_angle_of_rotation2(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi / 2
    else:
        if a > c:
            return np.arctan(2 * b / (a - c)) / 2
        else:
            return np.pi / 2 + np.arctan(2 * b / (a - c)) / 2


def fit_line(x, y, image):
    if x[0] == x[1]:
        x[0] += 0.1
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(0, image.shape[1], 50)
    y_axis = polynomial(x_axis)
    eye_line = []
    for i in range(len(x_axis)):
        eye_line.append((x_axis[i], y_axis[i]))

    return eye_line


def get_face_ellipse(face_landmark):
    chin = face_landmark["chin"]
    x = []
    y = []
    for point in chin:
        x.append(point[0])
        y.append(point[1])

    x = np.asarray(x)
    y = np.asarray(y)

    a = fitEllipse(x, y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    a, b = axes

    arc = 2.2
    R = np.arange(0, arc * np.pi, 0.2)
    xx = center[0] + a * np.cos(R) * np.cos(phi) - b * np.sin(R) * np.sin(phi)
    yy = center[1] + a * np.cos(R) * np.sin(phi) + b * np.sin(R) * np.cos(phi)
    chin_extrapolated = []
    for i in range(len(R)):
        chin_extrapolated.append((xx[i], yy[i]))
    face_landmark["chin_extrapolated"] = chin_extrapolated
    return face_landmark


def get_line(face_landmark, image, type="eye", debug=False):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    left_eye = face_landmark["left_eye"]
    right_eye = face_landmark["right_eye"]
    left_eye_mid = np.mean(np.array(left_eye), axis=0)
    right_eye_mid = np.mean(np.array(right_eye), axis=0)
    eye_line_mid = (left_eye_mid + right_eye_mid) / 2

    if type == "eye":
        left_point = left_eye_mid
        right_point = right_eye_mid
        mid_point = eye_line_mid

    elif type == "nose_mid":
        nose_length = (
                face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length / 2]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length / 2]

        mid_pointY = (
                             face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
                     ) / 2
        mid_pointX = (
                             face_landmark["nose_bridge"][-1][0] + face_landmark["nose_bridge"][0][0]
                     ) / 2
        mid_point = (mid_pointX, mid_pointY)

    elif type == "nose_tip":
        nose_length = (
                face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length]
        mid_point = (
                            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
                    ) / 2

    elif type == "bottom_lip":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.max(np.array(bottom_lip), axis=0)
        shiftY = bottom_lip_mid[1] - eye_line_mid[1]
        left_point = [left_eye_mid[0], left_eye_mid[1] + shiftY]
        right_point = [right_eye_mid[0], right_eye_mid[1] + shiftY]
        mid_point = bottom_lip_mid

    elif type == "perp_line":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.mean(np.array(bottom_lip), axis=0)

        left_point = eye_line_mid
        left_point = face_landmark["nose_bridge"][0]
        right_point = bottom_lip_mid

        mid_point = bottom_lip_mid

    elif type == "nose_long":
        nose_bridge = face_landmark["nose_bridge"]
        left_point = [nose_bridge[0][0], nose_bridge[0][1]]
        right_point = [nose_bridge[-1][0], nose_bridge[-1][1]]

        mid_point = left_point

    y = [left_point[1], right_point[1]]
    x = [left_point[0], right_point[0]]

    eye_line = fit_line(x, y, image)
    d.line(eye_line, width=5, fill="blue")

    # Perpendicular Line
    y = [
        (left_point[1] + right_point[1]) / 2,
        (left_point[1] + right_point[1]) / 2 + right_point[0] - left_point[0],
    ]
    x = [
        (left_point[0] + right_point[0]) / 2,
        (left_point[0] + right_point[0]) / 2 - right_point[1] + left_point[1],
    ]
    perp_line = fit_line(x, y, image)
    if debug:
        d.line(perp_line, width=5, fill="red")
        pil_image.show()
    return eye_line, perp_line, left_point, right_point, mid_point


def line_intersection(line1, line2):
    # mid = int(len(line1) / 2)
    start = 0
    end = -1
    line1 = ([line1[start][0], line1[start][1]], [line1[end][0], line1[end][1]])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    x = []
    y = []
    flag = False

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return flag, x, y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    segment_minX = min(line2[0][0], line2[1][0])
    segment_maxX = max(line2[0][0], line2[1][0])

    segment_minY = min(line2[0][1], line2[1][1])
    segment_maxY = max(line2[0][1], line2[1][1])

    if (
            segment_maxX + 1 >= x >= segment_minX - 1
            and segment_maxY + 1 >= y >= segment_minY - 1
    ):
        flag = True

    return flag, x, y


def get_angle(line1, line2):
    delta_y = line1[-1][1] - line1[0][1]
    delta_x = line1[-1][0] - line1[0][0]
    perp_angle = math.degrees(math.atan2(delta_y, delta_x))
    if delta_x < 0:
        perp_angle = perp_angle + 180
    if perp_angle < 0:
        perp_angle += 360
    if perp_angle > 180:
        perp_angle -= 180

    # print("perp", perp_angle)
    delta_y = line2[-1][1] - line2[0][1]
    delta_x = line2[-1][0] - line2[0][0]
    nose_angle = math.degrees(math.atan2(delta_y, delta_x))

    if delta_x < 0:
        nose_angle = nose_angle + 180
    if nose_angle < 0:
        nose_angle += 360
    if nose_angle > 180:
        nose_angle -= 180
    # print("nose", nose_angle)

    angle = nose_angle - perp_angle
    return angle


def get_points_on_chin(line, face_landmark, chin_type="chin"):
    chin = face_landmark[chin_type]
    points_on_chin = []
    for i in range(len(chin) - 1):
        chin_first_point = [chin[i][0], chin[i][1]]
        chin_second_point = [chin[i + 1][0], chin[i + 1][1]]

        flag, x, y = line_intersection(line, (chin_first_point, chin_second_point))
        if flag:
            points_on_chin.append((x, y))

    return points_on_chin


def get_six_points(face_landmark, image):
    _, perp_line1, _, _, m = get_line(face_landmark, image, type="nose_mid")
    face_b = m

    perp_line, _, _, _, _ = get_line(face_landmark, image, type="perp_line")
    points1 = get_points_on_chin(perp_line1, face_landmark)
    points = get_points_on_chin(perp_line, face_landmark)
    if not points1:
        face_e = tuple(np.asarray(points[0]))
    elif not points:
        face_e = tuple(np.asarray(points1[0]))
    else:
        face_e = tuple((np.asarray(points[0]) + np.asarray(points1[0])) / 2)
    # face_e = points1[0]
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_long")

    angle = get_angle(perp_line, nose_mid_line)
    # print("angle: ", angle)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_tip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    if len(points) < 2:
        face_landmark = get_face_ellipse(face_landmark)
        # print("extrapolating chin")
        points = get_points_on_chin(
            nose_mid_line, face_landmark, chin_type="chin_extrapolated"
        )
        if len(points) < 2:
            points = []
            points.append(face_landmark["chin"][0])
            points.append(face_landmark["chin"][-1])
    face_a = points[0]
    face_c = points[-1]
    # cv2.imshow('j', image)
    # cv2.waitKey(0)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="bottom_lip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    face_d = points[0]
    face_f = points[-1]

    six_points = np.float32([face_a, face_b, face_c, face_f, face_e, face_d])

    return six_points, angle


def parse_face(bbox):
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = xmin + w
    ymax = ymin + h
    return xmin, ymin, xmax, ymax  # top, left, bottom, right


def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points


def estimate_averaged_yaw(landmarks):
    # Works much better than solvePnP if landmarks from "3DFAN"
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    l = ((landmarks[27][0] - landmarks[0][0]) + (landmarks[28][0] - landmarks[1][0]) + (
            landmarks[29][0] - landmarks[2][0])) / 3.0
    r = ((landmarks[16][0] - landmarks[27][0]) + (landmarks[15][0] - landmarks[28][0]) + (
            landmarks[14][0] - landmarks[29][0])) / 3.0
    return float(r - l)


def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def ConvertIfStringIsInt(input_string):
    try:
        float(input_string)

        try:
            if int(input_string) == float(input_string):
                return int(input_string)
            else:
                return float(input_string)
        except ValueError:
            return float(input_string)

    except ValueError:
        return input_string


def read_cfg(config_filename="./Pluggins/face_generate_mask/masks.cfg", mask_type="surgical", verbose=False):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_filename)
    cfg = DotMap()
    section_name = mask_type

    if verbose:
        hyphens = "-" * int((80 - len(config_filename)) / 2)
        print(hyphens + " " + config_filename + " " + hyphens)

    # for section_name in parser.sections():

    if verbose:
        print("[" + section_name + "]")
    for name, value in parser.items(section_name):
        value = ConvertIfStringIsInt(value)
        if name != "template":
            cfg[name] = tuple(int(s) for s in value.split(","))
        else:
            cfg[name] = value
        spaces = " " * (30 - len(name))
        if verbose:
            print(name + ":" + spaces + str(cfg[name]))

    return cfg


def mask_face(image, face_location, six_points, angle, args, mask_type="surgical"):
    debug = False

    # Find the face angle
    threshold = 13
    if angle < -threshold:
        mask_type += "_right"
    elif angle > threshold:
        mask_type += "_left"

    face_height = face_location[2] - face_location[0]
    face_width = face_location[1] - face_location[3]

    # Read appropriate mask image
    w = image.shape[0]
    h = image.shape[1]
    if not "empty" in mask_type and not "inpaint" in mask_type:
        cfg = read_cfg(config_filename="./Pluggins/face_generate_mask/masks.cfg",
                       mask_type=mask_type, verbose=False)
    else:
        if "left" in mask_type:
            mask_str = "surgical_blue_left"
        elif "right" in mask_type:
            mask_str = "surgical_blue_right"
        else:
            mask_str = "surgical_blue"
        cfg = read_cfg(config_filename="./Pluggins/face_generate_mask/masks.cfg", mask_type=mask_str, verbose=False)

    img = cv2.imread(cfg.template, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    #     # Process the mask if necessary
    #     if args["pattern"]:
    #         # Apply pattern to mask
    #         img = texture_the_mask(img, args["pattern"], args["pattern_weight"])

    #     if args["color"]:
    #         # Apply color to mask
    #         img = color_the_mask(img, args["color"], args["color_weight"])

    mask_line = np.float32(
        [cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d]
    )
    # Warp the mask
    M, mask = cv2.findHomography(mask_line, six_points)
    dst_mask = cv2.warpPerspective(img, M, (h, w))
    dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)
    mask = dst_mask[:, :, 3]
    face_height = face_location[2] - face_location[0]
    face_width = face_location[1] - face_location[3]
    image_face = image[
                 face_location[0] + int(face_height / 2): face_location[2],
                 face_location[3]: face_location[1],
                 :,
                 ]

    image_face = image

    # Adjust Brightness
    mask_brightness = get_avg_brightness(img)
    img_brightness = get_avg_brightness(image_face)
    delta_b = 1 + (img_brightness - mask_brightness) / 255
    dst_mask = change_brightness(dst_mask, delta_b)

    # Adjust Saturation
    mask_saturation = get_avg_saturation(img)
    img_saturation = get_avg_saturation(image_face)
    delta_s = 1 - (img_saturation - mask_saturation) / 255
    dst_mask = change_saturation(dst_mask, delta_s)

    # Apply mask
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask)
    out_img = cv2.add(img_bg, img_fg[:, :, 0:3])
    if "empty" in mask_type or "inpaint" in mask_type:
        out_img = img_bg
    # Plot key points

    if "inpaint" in mask_type:
        out_img = cv2.inpaint(out_img, mask, 3, cv2.INPAINT_TELEA)

    if debug:
        for i in six_points:
            cv2.circle(out_img, (i[0], i[1]), radius=4, color=(0, 0, 255), thickness=-1)

        for i in dst_mask_points:
            cv2.circle(
                out_img, (i[0][0], i[0][1]), radius=4, color=(0, 255, 0), thickness=-1
            )

    return out_img, mask
