"""Астрофотостекинг в духе Sequator: дарк-кадры, выравнивание по звёздам,
заморозка земли, отбраковка шума, пост-обработка."""

import cv2
import numpy as np

from .common import (
    align_sky_affine,
    align_translation,
    load_raw,
    to_gray,
    warp_image,
)

MODES = ("average", "median", "maximum", "sigma_clip")
MODE_LABELS_RU = {
    "average": "Среднее",
    "median": "Медиана",
    "maximum": "Максимум",
    "sigma_clip": "Отбраковка (сигма-клип)",
}

_BAND_ROWS = 128


def stack_astro(
    light_paths,
    dark_paths=None,
    align=True,
    horizon=0.5,
    mode="sigma_clip",
    sigma_k=3.0,
    exposure=1.0,
    saturation=1.15,
    chroma_nr=True,
    progress=None,
):
    """Стекинг кадров ночного неба.

    light_paths — светлые кадры (первый — эталон);
    dark_paths — тёмные кадры для вычитания шума матрицы (опционально);
    align — выравнивание по звёздам (только небо, земля берётся с эталона);
    horizon — доля высоты кадра: всё выше — небо, ниже — земля (0..1);
    mode — 'average' | 'median' | 'maximum' | 'sigma_clip';
    sigma_k — порог отбраковки в сигмах для режима 'sigma_clip';
    exposure — множитель яркости; saturation — множитель насыщенности;
    chroma_nr — сглаживание цветного шума.
    """

    def report(frac, msg):
        if progress:
            progress(frac, msg)

    if len(light_paths) < 2:
        raise ValueError("Нужно минимум 2 светлых кадра")
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим стекинга: {mode}")

    report(0.02, "Загрузка светлых кадров")
    frames = [load_raw(p) for p in light_paths]

    h, w = frames[0][0].shape[:2]
    for i, (arr, _) in enumerate(frames):
        if arr.shape[:2] != (h, w):
            raise ValueError(
                f"Кадр {i + 1} имеет размер {arr.shape[1]}x{arr.shape[0]}, "
                f"ожидается {w}x{h}"
            )
    scale = frames[0][1]

    dark = _master_dark(dark_paths, (h, w)) if dark_paths else None

    sky_mask = np.zeros((h, w), dtype=np.float32)
    sky_row = int(np.clip(horizon, 0.0, 1.0) * h)
    sky_mask[:sky_row] = 1.0

    ref, _ = frames[0]
    ref_f = ref.astype(np.float32) / scale
    aligned = [ref_f]

    n = len(frames)
    if align:
        ref_gray = to_gray(ref_f)
        for i in range(1, n):
            report(0.1 + 0.45 * i / n, f"Выравнивание по звёздам: кадр {i + 1}/{n}")
            img_f = frames[i][0].astype(np.float32) / frames[i][1]
            matrix = align_sky_affine(ref_gray, to_gray(img_f), sky_mask)
            if matrix is None:
                matrix = align_translation(ref_gray, to_gray(img_f), scale=0.25)
            if matrix is None:
                aligned.append(img_f)
                continue
            warped = warp_image(img_f, matrix)
            wmask = warp_image(sky_mask, matrix)[..., None]
            aligned.append(warped * wmask + ref_f * (1.0 - wmask))
    else:
        for i in range(1, n):
            aligned.append(frames[i][0].astype(np.float32) / frames[i][1])

    report(0.6, f"Стекинг ({MODE_LABELS_RU[mode]})")
    stacked = _stack_pixels(aligned, dark, mode, sigma_k, report)

    report(0.9, "Пост-обработка")
    result = np.clip(stacked * float(exposure), 0.0, 1.0)
    if chroma_nr:
        result = _reduce_chroma_noise(result)
    if saturation != 1.0:
        luma = to_gray(result)[..., None]
        result = np.clip(luma + (result - luma) * float(saturation), 0.0, 1.0)

    report(1.0, "Готово")
    return result


def _master_dark(dark_paths, size):
    if not dark_paths:
        return None
    darks = []
    for p in dark_paths:
        arr, sc = load_raw(p)
        if arr.shape[:2] != size:
            continue
        darks.append(arr.astype(np.float32) / sc)
    if not darks:
        return None
    return np.median(np.stack(darks), axis=0)


def _stack_pixels(frames, dark, mode, sigma_k, report):
    h = frames[0].shape[0]
    w = frames[0].shape[1]
    out = np.empty((h, w, 3), dtype=np.float32)
    y = 0
    step = max(1, h // 20)
    while y < h:
        y1 = min(h, y + _BAND_ROWS)
        band = np.stack([f[y:y1] for f in frames])
        if dark is not None:
            band = np.maximum(band - dark[y:y1][None], 0.0)
        if mode == "average":
            out[y:y1] = band.mean(axis=0)
        elif mode == "maximum":
            out[y:y1] = band.max(axis=0)
        elif mode == "median":
            out[y:y1] = np.median(band, axis=0)
        else:
            out[y:y1] = _sigma_clip_mean(band, sigma_k)
        if y // _BAND_ROWS % step == 0:
            report(0.6 + 0.28 * (y1 / h), "Стекинг полос изображения")
        y = y1
    return out


def _sigma_clip_mean(band, sigma_k):
    med = np.median(band, axis=0)
    mad = np.median(np.abs(band - med), axis=0)
    thresh = sigma_k * (1.4826 * mad) + 1e-4
    keep = np.abs(band - med) <= thresh
    keep = np.maximum(keep, (keep.sum(axis=0) == 0)[None])
    count = keep.sum(axis=0).astype(np.float32)
    return (band * keep).sum(axis=0) / np.maximum(count, 1.0)


def _reduce_chroma_noise(img):
    ycc = cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_BGR2YCrCb)
    ycc[..., 1] = cv2.GaussianBlur(ycc[..., 1], (5, 5), 1.5)
    ycc[..., 2] = cv2.GaussianBlur(ycc[..., 2], (5, 5), 1.5)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
