"""Фокус-стекинг для пейзажа + тональные пресеты «Золотой час» / «Синий час»."""

import cv2
import numpy as np

from .common import (
    align_translation,
    gaussian_pyramid,
    laplacian_pyramid,
    load_image,
    pyramid_levels,
    to_gray,
    warp_image,
)

PRESETS = ("none", "golden_hour", "blue_hour")


def apply_preset(img, preset):
    """Применяет тональный пресет к float32 BGR [0, 1]."""
    if preset == "golden_hour":
        img = img * np.float32([0.86, 0.99, 1.14])
        img = _s_curve(img, 0.35)
        img = _saturate(img, 1.18)
    elif preset == "blue_hour":
        img = img * np.float32([1.12, 1.00, 0.88])
        img = img * 0.88 + 0.05
        img = _s_curve(img, 0.22)
        img = _saturate(img, 1.10)
    return np.clip(img, 0.0, 1.0)


def _s_curve(img, amount):
    smooth = img * img * (3.0 - 2.0 * img)
    return img * (1.0 - amount) + smooth * amount


def _saturate(img, factor):
    luma = to_gray(img)[..., None]
    return luma + (img - luma) * factor


def _focus_weight(img, power):
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    measure = cv2.blur(np.abs(lap), (7, 7))
    return measure ** power


def focus_stack(paths, progress=None, detail=1.0, preset="none"):
    """Многофокусный стекинг: выравнивание + лапласиева пирамида с весами резкости.

    paths — список путей (первый кадр считается эталонным);
    detail — степень влияния карты резкости (0.5–3.0);
    preset — 'none' | 'golden_hour' | 'blue_hour'.
    """
    if len(paths) < 2:
        raise ValueError("Нужно минимум 2 кадра для стекинга")

    def report(frac, msg):
        if progress:
            progress(frac, msg)

    ref = load_image(paths[0])
    h, w = ref.shape[:2]
    levels = pyramid_levels(ref.shape)
    report(0.0, f"Кадров: {len(paths)}, уровней пирамиды: {levels}")

    acc_lp = [None] * (levels + 1)
    acc_wp = [None] * (levels + 1)

    n = len(paths)
    for i, path in enumerate(paths):
        report(i / n, f"Кадр {i + 1}/{n}: загрузка и выравнивание")
        img = ref if i == 0 else _aligned_frame(ref, path, h, w)
        report((i + 0.35) / n, f"Кадр {i + 1}/{n}: карта резкости и смешивание")
        weight = _focus_weight(img, detail)

        lp, low = laplacian_pyramid(img, levels)
        wp = gaussian_pyramid(weight, levels)

        for lvl in range(levels):
            acc_lp[lvl] = _accumulate(acc_lp[lvl], lp[lvl], wp[lvl])
            acc_wp[lvl] = wp[lvl] if acc_wp[lvl] is None else acc_wp[lvl] + wp[lvl]
        acc_lp[levels] = _accumulate(acc_lp[levels], low, wp[levels])
        acc_wp[levels] = wp[levels] if acc_wp[levels] is None else acc_wp[levels] + wp[levels]

        report((i + 0.6) / n, f"Кадр {i + 1}/{n}: готово")

    report(0.9, "Сборка пирамиды")
    low = acc_lp[levels] / np.maximum(acc_wp[levels], 1e-6)[..., None]
    for lvl in reversed(range(levels)):
        size = (acc_lp[lvl].shape[1], acc_lp[lvl].shape[0])
        low = cv2.pyrUp(low, dstsize=size) + acc_lp[lvl] / np.maximum(acc_wp[lvl], 1e-6)[..., None]

    result = np.clip(low, 0.0, 1.0)
    report(0.95, "Пресет тона")
    result = apply_preset(result, preset)
    report(1.0, "Готово")
    return result


def _accumulate(acc, value, weight):
    weighted = value * weight[..., None]
    if acc is None:
        return weighted.astype(np.float32)
    acc += weighted
    return acc


def _aligned_frame(ref, path, h, w):
    img = load_image(path)
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    matrix = align_translation(to_gray(ref), to_gray(img), scale=0.25)
    if matrix is not None:
        img = warp_image(img, matrix)
    return img
