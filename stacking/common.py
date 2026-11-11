"""Общие вспомогательные функции: загрузка/сохранение, выравнивание, пирамиды."""

from pathlib import Path

import cv2
import numpy as np

ECC_TERM = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)


def load_image(path):
    """Загружает изображение как float32 BGR в диапазоне [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не удалось прочитать файл: {path}")
    return img.astype(np.float32) / 255.0


def load_raw(path):
    """Загружает изображение без преобразования (uint8/uint16) для экономии памяти.

    Возвращает кортеж (массив BGR, делитель для приведения к [0, 1]).
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Не удалось прочитать файл: {path}")
    if img.dtype == np.uint16:
        return img, np.float32(65535.0)
    return img.astype(np.uint8), np.float32(255.0)


def save_image(path, img):
    """Сохраняет float32 BGR [0, 1]. PNG/TIFF пишутся в 16 бит, JPEG — 8 бит."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0.0, 1.0)
    ext = path.suffix.lower()
    if ext in (".png", ".tif", ".tiff"):
        out = np.round(img * 65535.0).astype(np.uint16)
        if not cv2.imwrite(str(path), out):
            raise IOError(f"Не удалось сохранить: {path}")
    else:
        out = np.round(img * 255.0).astype(np.uint8)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
        if not cv2.imwrite(str(path), out, params):
            raise IOError(f"Не удалось сохранить: {path}")


def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def align_translation(ref_gray, img_gray, scale=0.25):
    """Оценивает сдвиг относительно эталона (ECC на уменьшенной копии).

    Возвращает матрицу 2x3 для полного разрешения или None при неудаче.
    """
    h, w = ref_gray.shape
    small = (int(round(w * scale)), int(round(h * scale)))
    ref_s = cv2.resize(ref_gray, small, interpolation=cv2.INTER_AREA)
    img_s = cv2.resize(img_gray, small, interpolation=cv2.INTER_AREA)
    ref_s = cv2.GaussianBlur(ref_s, (5, 5), 0)
    img_s = cv2.GaussianBlur(img_s, (5, 5), 0)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            ref_s, img_s, warp, cv2.MOTION_TRANSLATION, ECC_TERM, None, 5
        )
    except cv2.error:
        return None
    warp = warp.astype(np.float32)
    warp[:, 2] /= np.float32(scale)
    return warp


def to_gray_u8(gray):
    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)


def align_sky_affine(ref_gray, img_gray, sky_mask):
    """Выравнивание по звёздам: ORB только по небу + RANSAC (частичное аффинное).

    Возвращает матрицу 2x3 или None, если звёзд мало.
    """
    mask = (sky_mask * 255).astype(np.uint8)
    ref_u8 = to_gray_u8(ref_gray)
    img_u8 = to_gray_u8(img_gray)
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=5)
    kp1, des1 = orb.detectAndCompute(ref_u8, mask)
    kp2, des2 = orb.detectAndCompute(img_u8, mask)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des2, des1, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 8:
        return None
    return matrix.astype(np.float32)


def warp_image(img, matrix):
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


def gaussian_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def laplacian_pyramid(img, levels):
    gp = gaussian_pyramid(img, levels)
    lp = [
        gp[i] - cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        for i in range(levels)
    ]
    return lp, gp[-1]


def pyramid_levels(shape, smallest=64, lo=2, hi=6):
    min_dim = min(shape[:2])
    if min_dim <= smallest:
        return lo
    return int(np.clip(int(np.log2(min_dim / smallest)), lo, hi))
