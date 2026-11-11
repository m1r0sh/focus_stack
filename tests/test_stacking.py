"""Headless-тесты стекинга на синтетических данных. Запуск: python tests/test_stacking.py"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stacking.astro import stack_astro
from stacking.common import align_sky_affine, save_image
from stacking.landscape import apply_preset, focus_stack


def make_scene(size=(480, 640)):
    h, w = size
    img = np.zeros((h, w, 3), np.float32)
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img[..., 0] = 0.1 + 0.3 * yy
    img[..., 1] = 0.2 + 0.4 * xx
    img[..., 2] = 0.3 * (xx * yy)
    cv2.circle(img, (w // 4, h // 3), 40, (0.9, 0.4, 0.1), -1)
    cv2.rectangle(img, (w // 2, h // 2), (w // 2 + 120, h // 2 + 80), (0.1, 0.8, 0.2), -1)
    for k in range(6):
        cv2.rectangle(img, (w // 3 + k * 18, int(h * 0.75)), (w // 3 + k * 18 + 8, int(h * 0.85)), (1, 1, 1), -1)
    return img


def test_landscape_focus_stack():
    scene = make_scene()
    h, w = scene.shape[:2]
    top_sharp = scene.copy()
    top_sharp[h // 2 :] = cv2.GaussianBlur(top_sharp[h // 2 :], (0, 0), 6.0)
    bottom_sharp = scene.copy()
    bottom_sharp[: h // 2] = cv2.GaussianBlur(bottom_sharp[: h // 2], (0, 0), 6.0)

    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = os.path.join(tmp, "a.png"), os.path.join(tmp, "b.png")
        save_image(p1, top_sharp)
        save_image(p2, bottom_sharp)
        out = focus_stack([p1, p2], detail=1.0)

    err = float(np.mean(np.abs(out - scene)))
    sharp_out = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    sharp_in = cv2.Laplacian(cv2.cvtColor(top_sharp, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    assert err < 0.03, f"Отклонение от эталона слишком велико: {err:.4f}"
    assert sharp_out > sharp_in * 0.8, f"Результат менее резкий, чем вход: {sharp_out:.1f} vs {sharp_in:.1f}"
    print(f"[OK] landscape focus stack: err={err:.4f}, sharpness={sharp_out:.1f}")


def test_presets():
    img = np.full((120, 160, 3), 0.4, dtype=np.float32)
    golden = apply_preset(img.copy(), "golden_hour")
    blue = apply_preset(img.copy(), "blue_hour")
    ratio0 = img[..., 2].mean() / img[..., 0].mean()
    ratio_golden = golden[..., 2].mean() / golden[..., 0].mean()
    ratio_blue = blue[..., 2].mean() / blue[..., 0].mean()
    assert ratio_golden > ratio0 * 1.2, "Золотой час должен быть теплее исходника"
    assert ratio_blue < ratio0 * 0.85, "Синий час должен быть холоднее исходника"
    assert golden.max() <= 1.0 and blue.min() >= 0.0
    print(f"[OK] presets: R/B base={ratio0:.2f} golden={ratio_golden:.2f} blue={ratio_blue:.2f}")


def make_sky(size=(480, 640), seed=0, n_stars=250):
    rng = np.random.default_rng(seed)
    h, w = size
    img = rng.normal(0.03, 0.015, (h, w, 3)).astype(np.float32)
    img[..., 0] *= 0.8
    img[..., 2] *= 1.2
    positions = rng.uniform([20, 20], [h - 20, w - 20], (n_stars, 2))
    for x, y in positions:
        cv2.circle(img, (int(x), int(y)), 1 + int(rng.integers(0, 2)), (1.0, 1.0, 1.0), -1)
    return img, positions


def test_astro_alignment_and_stack():
    base, _ = make_sky(seed=42)
    h, w = base.shape[:2]
    horizon = 0.6
    shifts = [(0, 0), (7, -5), (-6, 4), (3, 9)]

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (dx, dy) in enumerate(shifts):
            frame = np.roll(base, (dy, dx), axis=(0, 1))
            frame = np.clip(frame + np.random.default_rng(i).normal(0, 0.02, frame.shape), 0, 1)
            p = os.path.join(tmp, f"light_{i}.png")
            save_image(p, frame)
            paths.append(p)

        single = cv2.imread(paths[0]).astype(np.float32) / 255
        out = stack_astro(
            paths, align=True, horizon=horizon, mode="sigma_clip", sigma_k=3.0,
            exposure=1.0, saturation=1.0, chroma_nr=False,
        )

    sky = slice(0, int(horizon * h) - 20)
    noise_single = float(np.median(np.std(single[sky], axis=2)))
    noise_stack = float(np.median(np.std(out[sky], axis=2)))
    assert noise_stack < noise_single * 0.7, (
        f"Шум не уменьшился: {noise_stack:.4f} vs {noise_single:.4f}"
    )
    print(f"[OK] astro stack: background noise {noise_single:.4f} -> {noise_stack:.4f}")


def test_sky_affine_estimator():
    base, _ = make_sky(seed=7)
    h, w = base.shape[:2]
    mask = np.zeros((h, w), np.float32)
    mask[: int(0.6 * h)] = 1.0
    shift = np.float32([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    moved = cv2.warpAffine(base, shift, (w, h))
    est = align_sky_affine(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY), mask)
    assert est is not None, "Выравнивание не нашло матрицу"
    expected = np.float32([[1.0, 0.0, -5.0], [0.0, 1.0, 3.0]])
    err = float(np.abs(est - expected).max())
    assert err < 1.5, f"Матрица далека от истинной: {est} (err={err:.2f})"
    print(f"[OK] sky affine: err={err:.3f}, M={est.tolist()}")


def test_astro_modes():
    base, _ = make_sky(seed=3, n_stars=120)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(4):
            frame = np.clip(base + np.random.default_rng(i).normal(0, 0.05, base.shape), 0, 1)
            p = os.path.join(tmp, f"f_{i}.png")
            save_image(p, frame)
            paths.append(p)
        for mode in ("average", "median", "maximum", "sigma_clip"):
            out = stack_astro(paths, align=False, mode=mode, chroma_nr=False)
            assert out.shape == base.shape and out.dtype == np.float32
            assert 0.0 <= out.min() and out.max() <= 1.0
            print(f"[OK] astro mode '{mode}': min={out.min():.3f} max={out.max():.3f}")


if __name__ == "__main__":
    test_landscape_focus_stack()
    test_presets()
    test_sky_affine_estimator()
    test_astro_alignment_and_stack()
    test_astro_modes()
    print("\nВсе тесты пройдены.")
