# Completed by Muse 2026-11-10 (Round 5).
#
# Honest origin note -- read before using this file:
# The original file in this repo ("python nexus_agi_ultimate_system.py")
# was a truncated copy of a Google Research EfficientDet inference
# utility, cut off mid-dict at line 561 inside ServingDriver.load()
# ("'image_file" unterminated), carrying a 2020 Google Research
# copyright header. That upstream code is NOT reproduced here, and
# the Google copyright header has been removed entirely.
#
# This file is a NEW, self-contained inference helper written for this
# repo by Muse, inspired only by the fragment's intent: image
# preprocessing, batched inference, detection post-processing
# (non-maximum suppression), visualization, and a serving driver. It
# is not Google's code and does not claim EfficientDet compatibility,
# TensorFlow compatibility, or any upstream behaviour. The public
# surface the fragment promised (image_preprocess,
# batch_image_files_decode, batch_image_preprocess, build_inputs,
# build_model, restore_ckpt, det_post_process, visualize_image,
# visualize_image_prediction, ServingDriver) is kept so anything that
# imported the fragment gets what it expected -- but the model
# dependent parts are honest no-backend stubs (see
# NoDetectionBackendError): no detection model is bundled with this
# repo, and none is faked. Supply your own `model_fn` callable to run
# real inference.
#
# Dependencies: Python standard library + numpy. Pillow is optional:
# image file decoding and drawing need it; if Pillow is not installed
# the module still imports, decode/visualization raise or degrade with
# a clear message instead of crashing at import time.
#
# This module makes no consciousness, sentience, or AGI claims. It is
# a plain image-preprocessing / detection-post-processing helper.
"""Self-contained inference helper (no model backend bundled).

Written from scratch by Muse (Round 5, 2026-11-10) to replace a
truncated fragment. Public surface:

  image_preprocess, batch_image_files_decode, batch_image_preprocess,
  build_inputs, build_model, restore_ckpt, det_post_process,
  visualize_image, visualize_image_prediction, ServingDriver.

Real, dependency-free implementations: preprocessing, batched
preprocessing, and pure-numpy non-maximum suppression post-processing.
Honest stubs (raise NoDetectionBackendError): build_model,
restore_ckpt, and the ServingDriver model paths -- unless the caller
supplies a model_fn / loader callable, in which case they are wired
in and genuinely run.

Run `python nexus_agi_ultimate_system.py --demo` for a model-free
smoke test (synthetic image, fake detections, NMS proof, and the
no-backend error shown, not raised).
"""

import glob
import logging
import os
import time

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:  # Pillow is optional; numpy-only paths still work.
    Image = None
    ImageDraw = None
    ImageFont = None
    _PIL_AVAILABLE = False

__all__ = [
    "NoDetectionBackendError",
    "image_preprocess",
    "batch_image_files_decode",
    "batch_image_preprocess",
    "build_inputs",
    "build_model",
    "restore_ckpt",
    "det_post_process",
    "visualize_image",
    "visualize_image_prediction",
    "ServingDriver",
]

logger = logging.getLogger(__name__)


class NoDetectionBackendError(NotImplementedError):
    """Raised when a model-backed operation is attempted with no backend.

    This repo bundles no detection model and no TensorFlow. The
    operation is not silently faked: pass a `model_fn` callable (for
    build_model / ServingDriver) or a `loader` callable (for
    restore_ckpt / ServingDriver.load) to plug a real backend in.
    """


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _parse_image_size(image_size):
    """Normalize an image size to (height, width)."""
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError(
                "image_size as a tuple/list must be (height, width), "
                "got %r" % (image_size,))
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)


def _parse_mean_std(value, name):
    """Normalize a mean/stddev value to a (3,) float array.

    Accepts a scalar (applied to all channels) or a sequence of one
    value per channel.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.repeat(arr, 3)
    if arr.ndim == 1 and arr.shape[0] in (1, 3, 4):
        return arr
    raise ValueError(
        "%s must be a scalar or a per-channel sequence, got %r"
        % (name, value))


def _to_numpy_image(image):
    """Convert a numpy array or PIL image to a (H, W, C) numpy array."""
    if _PIL_AVAILABLE and isinstance(image, Image.Image):
        return np.array(image)
    arr = np.asarray(image)
    if arr.ndim == 2:  # Grayscale -> replicate to 3 channels.
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(
            "image must be a (H, W, C) array with C in (1, 3, 4) or a PIL "
            "image, got shape %s" % (arr.shape,))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 4:  # Drop alpha for the 3-channel pipeline.
        arr = arr[:, :, :3]
    return arr


def _bilinear_resize(image, target_h, target_w):
    """Resize a (H, W, C) float array to (target_h, target_w) bilinearly.

    Pure numpy; no image library needed. Aspect ratio is NOT preserved:
    the image is stretched to exactly the target size, matching the
    fragment's "resize and crop to the output size" contract in the
    simplest deterministic way.
    """
    src_h, src_w = image.shape[0], image.shape[1]
    if src_h == target_h and src_w == target_w:
        return image.astype(np.float64, copy=True)
    # Source coordinates of each target pixel centre (align-corners off).
    ys = (np.arange(target_h, dtype=np.float64) + 0.5) * src_h / target_h - 0.5
    xs = (np.arange(target_w, dtype=np.float64) + 0.5) * src_w / target_w - 0.5
    ys = np.clip(ys, 0.0, src_h - 1.0)
    xs = np.clip(xs, 0.0, src_w - 1.0)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = image[y0, :, :][:, x0, :] * (1.0 - wx) + image[y0, :, :][:, x1, :] * wx
    bottom = (image[y1, :, :][:, x0, :] * (1.0 - wx)
              + image[y1, :, :][:, x1, :] * wx)
    return top * (1.0 - wy) + bottom * wy


def _sigmoid(x):
    """Numerically stable sigmoid (numpy)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def image_preprocess(image, image_size, mean_rgb, stddev_rgb):
    """Preprocess an image for inference.

    Args:
      image: input image, a numpy array (H, W, C) or a PIL image.
      image_size: single integer of image size for a square image, or a
        tuple of two integers (image_height, image_width).
      mean_rgb: mean value(s) of RGB, a float or a list of floats.
      stddev_rgb: standard deviation(s) of RGB, a float or a list.

    Returns:
      (image, scale): the processed image as a float32 array of shape
      (target_h, target_w, 3), normalized as
      (pixel - mean_rgb) / stddev_rgb, and `scale`, a (scale_y, scale_x)
      tuple giving the factor to multiply preprocessed-image coordinates
      by to recover original-image coordinates.
    """
    target_h, target_w = _parse_image_size(image_size)
    mean = _parse_mean_std(mean_rgb, "mean_rgb")
    stddev = _parse_mean_std(stddev_rgb, "stddev_rgb")
    if np.any(stddev == 0):
        raise ValueError("stddev_rgb must not contain zero, got %r"
                         % (stddev_rgb,))

    arr = _to_numpy_image(image)
    orig_h, orig_w = arr.shape[0], arr.shape[1]
    resized = _bilinear_resize(arr.astype(np.float64), target_h, target_w)
    normalized = (resized - mean) / stddev
    scale = (orig_h / float(target_h), orig_w / float(target_w))
    return normalized.astype(np.float32), scale


def batch_image_files_decode(image_files):
    """Read image files from disk into a list of numpy arrays.

    Args:
      image_files: a list of file paths (or a single path string).

    Returns:
      A list of uint8 numpy arrays, one per file, each (H, W, 3).

    Raises:
      ImportError: if Pillow is not installed (decoding needs it).
      FileNotFoundError: if any listed file does not exist.
    """
    if isinstance(image_files, (str, bytes, os.PathLike)):
        image_files = [image_files]
    if not _PIL_AVAILABLE:
        raise ImportError(
            "batch_image_files_decode needs Pillow to read image files; "
            "install it (pip install Pillow) or pass numpy arrays to "
            "batch_image_preprocess directly.")
    raw_images = []
    for path in image_files:
        path = os.fspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError("image file not found: %s" % path)
        with Image.open(path) as img:
            raw_images.append(np.array(img.convert("RGB")))
    return raw_images


def batch_image_preprocess(raw_images, image_size, mean_rgb, stddev_rgb,
                           batch_size=None):
    """Preprocess a batch of images for inference.

    Args:
      raw_images: a list of images, each a numpy array or PIL image.
      image_size: single integer, or (height, width) tuple.
      mean_rgb: mean value(s) of RGB, a float or a list of floats.
      stddev_rgb: standard deviation(s) of RGB, a float or a list.
      batch_size: accepted for API compatibility with the fragment; if
        given it must equal len(raw_images). Ignored otherwise.

    Returns:
      (images, scales): `images` is a float32 array of shape
      (N, target_h, target_w, 3); `scales` is a list of N
      (scale_y, scale_x) tuples, one per image.
    """
    if batch_size is not None and int(batch_size) != len(raw_images):
        raise ValueError(
            "batch_size=%r does not match %d raw images"
            % (batch_size, len(raw_images)))
    images, scales = [], []
    for raw in raw_images:
        image, scale = image_preprocess(raw, image_size, mean_rgb, stddev_rgb)
        images.append(image)
        scales.append(scale)
    return np.stack(images).astype(np.float32), scales


def build_inputs(image_path_pattern, image_size, mean_rgb, stddev_rgb):
    """Read and preprocess input images matching a glob pattern.

    Args:
      image_path_pattern: a path or glob pattern for one or more files.

    Returns:
      (raw_images, images, scales): raw images as a list of uint8 numpy
      arrays, processed images as a float32 array, and a list of scales.

    Raises:
      ValueError: if the pattern matches no files.
      ImportError: if Pillow is not installed.
    """
    files = sorted(glob.glob(os.path.expanduser(image_path_pattern)))
    if not files:
        raise ValueError(
            "Cannot find any images for pattern %s" % image_path_pattern)
    raw_images = batch_image_files_decode(files)
    images, scales = batch_image_preprocess(raw_images, image_size,
                                            mean_rgb, stddev_rgb)
    return raw_images, images, scales


# ---------------------------------------------------------------------------
# Model handling -- honest no-backend stubs unless a backend is supplied
# ---------------------------------------------------------------------------

def build_model(model_name, inputs, model_fn=None, **kwargs):
    """Build a model for a given model name.

    This repo bundles no detection model, so with no `model_fn` this
    raises NoDetectionBackendError (a NotImplementedError) instead of
    pretending to build one.

    Args:
      model_name: the name of the model (kept for API compatibility;
        it is passed through to `model_fn`).
      inputs: preprocessed images, a numpy array of shape
        (batch, H, W, 3).
      model_fn: optional callable with signature
        ``model_fn(inputs, model_name=model_name, **kwargs)`` returning
        ``(cls_outputs, box_outputs)``. `cls_outputs` and `box_outputs`
        may be plain numpy arrays of shape (batch, N, num_classes) /
        (batch, N, 4), or dicts keyed by feature level as in the
        original fragment. See det_post_process for the formats.
      **kwargs: extra parameters forwarded to `model_fn`.

    Returns:
      (cls_outputs, box_outputs) as returned by `model_fn`.

    Raises:
      NoDetectionBackendError: if `model_fn` is None.
    """
    if model_fn is None:
        raise NoDetectionBackendError(
            "No detection model backend is bundled with this repo: "
            "build_model(%r) cannot build a model on its own. Pass a "
            "model_fn callable -- model_fn(inputs, model_name=%r, "
            "**kwargs) returning (cls_outputs, box_outputs) -- to plug "
            "in your own backend (for example a TensorFlow, PyTorch, "
            "or ONNX model). No model is faked here." % (model_name,
                                                        model_name))
    return model_fn(inputs, model_name=model_name, **kwargs)


def restore_ckpt(sess, ckpt_path, ema_decay=0.9998, export_ckpt=None,
                 loader=None):
    """Restore weights from a checkpoint path.

    No checkpoint format is bundled with this repo, so with no `loader`
    this validates the path and then raises NoDetectionBackendError
    instead of pretending to restore weights.

    Args:
      sess: kept for API compatibility with the fragment; ignored
        unless `loader` uses it.
      ckpt_path: path to a checkpoint file or directory. The special
        value ``'_'`` means "test mode: load nothing" (kept from the
        fragment) and returns None without error.
      ema_decay: kept for API compatibility; ignored without a backend.
      export_ckpt: kept for API compatibility; ignored without a
        backend unless `loader` handles it.
      loader: optional callable ``loader(ckpt_path, sess=sess,
        ema_decay=ema_decay, export_ckpt=export_ckpt)`` that performs
        the actual restore and returns whatever it restores.

    Returns:
      Whatever `loader` returns, or None for ckpt_path == '_'.

    Raises:
      FileNotFoundError: if `ckpt_path` (other than '_') does not exist.
      NoDetectionBackendError: if the path exists but no `loader` was
        supplied.
    """
    if ckpt_path == '_':
        logger.info("Running test: do not load any ckpt.")
        return None
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "checkpoint path does not exist: %s" % ckpt_path)
    if loader is None:
        raise NoDetectionBackendError(
            "No checkpoint backend is bundled with this repo: the path "
            "%r exists, but there is no loader that knows how to read "
            "it. Pass a loader callable -- loader(ckpt_path, sess=sess, "
            "ema_decay=..., export_ckpt=...) -- to plug in your own "
            "checkpoint format. No weights are faked here." % ckpt_path)
    return loader(ckpt_path, sess=sess, ema_decay=ema_decay,
                  export_ckpt=export_ckpt)


# ---------------------------------------------------------------------------
# Detection post-processing -- real pure-numpy NMS
# ---------------------------------------------------------------------------

def _flatten_level_outputs(cls_outputs, box_outputs, num_classes):
    """Flatten dict-of-level outputs to (batch, N, C) / (batch, N, 4).

    The fragment's format: cls level values are
    [batch, height, width, num_anchors * num_classes] and box level
    values are [batch, height, width, num_anchors * 4]. Each level is
    reshaped to (batch, height * width * num_anchors, ...) and the
    levels are concatenated.
    """
    cls_parts, box_parts = [], []
    for level in sorted(cls_outputs.keys()):
        cls = np.asarray(cls_outputs[level], dtype=np.float64)
        box = np.asarray(box_outputs[level], dtype=np.float64)
        if cls.ndim != 4 or box.ndim != 4:
            raise ValueError(
                "dict-of-level outputs must be 4-D [batch, H, W, ...]; "
                "level %r gave cls %s, box %s"
                % (level, cls.shape, box.shape))
        batch, h, w = cls.shape[0], cls.shape[1], cls.shape[2]
        anchors = cls.shape[3] // num_classes
        if anchors * num_classes != cls.shape[3]:
            raise ValueError(
                "level %r: cls last dim %d is not divisible by "
                "num_classes=%d" % (level, cls.shape[3], num_classes))
        if box.shape[3] % 4 != 0:
            raise ValueError(
                "level %r: box last dim %d is not divisible by 4"
                % (level, box.shape[3]))
        if box.shape[3] // 4 != anchors:
            raise ValueError(
                "level %r: cls implies %d anchors but box implies %d"
                % (level, anchors, box.shape[3] // 4))
        cls_parts.append(cls.reshape(batch, h * w * anchors, num_classes))
        box_parts.append(box.reshape(batch, h * w * anchors, 4))
    return (np.concatenate(cls_parts, axis=1),
            np.concatenate(box_parts, axis=1))


def _as_batched(cls_outputs, box_outputs, num_classes):
    """Normalize cls/box outputs to (batch, N, C) / (batch, N, 4)."""
    if isinstance(cls_outputs, dict) or isinstance(box_outputs, dict):
        if not (isinstance(cls_outputs, dict)
                and isinstance(box_outputs, dict)):
            raise ValueError(
                "cls_outputs and box_outputs must both be dicts or both "
                "be arrays, got %s and %s"
                % (type(cls_outputs).__name__, type(box_outputs).__name__))
        if num_classes is None:
            raise ValueError(
                "num_classes (in params) is required for dict-of-level "
                "outputs; it was not supplied")
        return _flatten_level_outputs(cls_outputs, box_outputs, num_classes)

    cls = np.asarray(cls_outputs, dtype=np.float64)
    box = np.asarray(box_outputs, dtype=np.float64)
    if cls.ndim == 2:  # Single image: (N, C).
        cls = cls[None, :, :]
    if box.ndim == 2:  # Single image: (N, 4).
        box = box[None, :, :]
    if cls.ndim != 3 or box.ndim != 3:
        raise ValueError(
            "cls_outputs must be (N, C) or (batch, N, C) and box_outputs "
            "(N, 4) or (batch, N, 4); got %s and %s"
            % (cls.shape, box.shape))
    if cls.shape[0] != box.shape[0] or cls.shape[1] != box.shape[1]:
        raise ValueError(
            "cls_outputs %s and box_outputs %s disagree on batch/anchor "
            "count" % (cls.shape, box.shape))
    if box.shape[2] != 4:
        raise ValueError(
            "box_outputs last dim must be 4 ([ymin, xmin, ymax, xmax]), "
            "got %s" % (box.shape,))
    return cls, box


def _iou(box_a, box_b):
    """IoU of two boxes in [ymin, xmin, ymax, xmax] format."""
    y_min = max(box_a[0], box_b[0])
    x_min = max(box_a[1], box_b[1])
    y_max = min(box_a[2], box_b[2])
    x_max = min(box_a[3], box_b[3])
    inter_h = max(0.0, y_max - y_min)
    inter_w = max(0.0, x_max - x_min)
    inter = inter_h * inter_w
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, scores, iou_thresh):
    """Greedy per-class NMS. Returns indices kept, best score first."""
    order = np.argsort(-scores)
    keep = []
    for idx in order:
        idx = int(idx)
        if all(_iou(boxes[idx], boxes[kept]) <= iou_thresh
               for kept in keep):
            keep.append(idx)
    return keep


def det_post_process(params, cls_outputs, box_outputs, scales):
    """Post-process box/class predictions with non-maximum suppression.

    Args:
      params: a parameter dictionary. Recognised keys:
        - ``num_classes``: required only for dict-of-level outputs.
        - ``min_score_thresh`` (default 0.05): minimum score to keep a
          detection. Also accepts ``score_thresh``.
        - ``nms_iou_thresh`` (default 0.5): two boxes of the same class
          with IoU above this are merged, keeping the higher score.
          Also accepts ``iou_thresh``.
        - ``max_detections`` (default 100): maximum detections kept
          per image.
        - ``image_size``: optional (height, width) of the preprocessed
          image; used only to document the box coordinate space.
      cls_outputs: class predictions, either
        - plain arrays: (N, num_classes) or (batch, N, num_classes) --
          per-anchor scores or logits (sigmoid is applied when values
          fall outside [0, 1]); or
        - dicts keyed by feature level, as in the original fragment:
          values [batch, H, W, num_anchors * num_classes].
      box_outputs: box predictions in [ymin, xmin, ymax, xmax] order,
        in preprocessed-image pixel coordinates, either
        - plain arrays: (N, 4) or (batch, N, 4); or
        - dicts keyed by feature level: values
          [batch, H, W, num_anchors * 4].
      scales: a list with one entry per image in the batch, giving the
        factor to rescale preprocessed-image coordinates back to
        original-image coordinates: a scalar (same for both axes) or a
        (scale_y, scale_x) tuple, as returned by image_preprocess.

    Returns:
      detections: a float64 numpy array of shape (total, 7); each row
      is [image_id, ymin, xmin, ymax, xmax, score, class] in
      original-image coordinates. Empty when nothing passes the
      thresholds (shape (0, 7)).
    """
    params = dict(params or {})
    score_thresh = float(params.get("min_score_thresh",
                                    params.get("score_thresh", 0.05)))
    iou_thresh = float(params.get("nms_iou_thresh",
                                  params.get("iou_thresh", 0.5)))
    max_detections = int(params.get("max_detections", 100))
    num_classes = params.get("num_classes")

    cls, box = _as_batched(cls_outputs, box_outputs, num_classes)

    # Logits (values outside [0, 1]) -> probabilities via sigmoid.
    if cls.size and (cls.min() < 0.0 or cls.max() > 1.0):
        cls = _sigmoid(cls)

    batch_size = cls.shape[0]
    if len(scales) != batch_size:
        raise ValueError(
            "scales has %d entries but the batch has %d images"
            % (len(scales), batch_size))

    rows = []
    for img_id in range(batch_size):
        scale = scales[img_id]
        if isinstance(scale, (tuple, list)):
            scale_y, scale_x = float(scale[0]), float(scale[1])
        else:
            scale_y = scale_x = float(scale)
        num_anchors = cls.shape[1]
        classes = cls.shape[2]
        kept = []
        for class_id in range(classes):
            scores = cls[img_id, :, class_id]
            cand = np.nonzero(scores >= score_thresh)[0]
            if cand.size == 0:
                continue
            for idx in _nms(box[img_id, cand], scores[cand], iou_thresh):
                anchor = int(cand[idx])
                kept.append((float(scores[anchor]), anchor, class_id))
        # Highest score first, then cap per image.
        kept.sort(key=lambda item: -item[0])
        kept = kept[:max_detections]
        for score, anchor, class_id in kept:
            ymin, xmin, ymax, xmax = box[img_id, anchor]
            rows.append([float(img_id),
                         float(ymin) * scale_y,
                         float(xmin) * scale_x,
                         float(ymax) * scale_y,
                         float(xmax) * scale_x,
                         float(score),
                         float(class_id)])
    if not rows:
        return np.zeros((0, 7), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _label_for(class_id, label_map):
    """Human-readable label for a class id."""
    if isinstance(label_map, dict):
        value = label_map.get(class_id, label_map.get(str(class_id)))
        if value is not None:
            return str(value)
    return "class_%d" % int(class_id)


def visualize_image(image, boxes, classes, scores, label_map=None,
                    min_score_thresh=0.01, max_boxes_to_draw=1000,
                    line_thickness=2, **kwargs):
    """Draw boxes and labels on an image.

    Args:
      image: an image with shape [H, W, C] (numpy array or PIL image).
      boxes: box predictions with shape [N, 4] in [ymin, xmin, ymax,
        xmax] order, in image pixel coordinates.
      classes: class predictions with shape [N].
      scores: scores with shape [N].
      label_map: a dict from class id to display name. Any other type
        (e.g. the fragment's 'coco' string) is ignored with a note:
        no label catalogue is bundled, so class ids are shown as
        "class_<id>".
      min_score_thresh: boxes below this score are not drawn.
      max_boxes_to_draw: maximum boxes to draw.
      line_thickness: box outline thickness in pixels.
      **kwargs: ignored (API compatibility).

    Returns:
      The annotated image as a uint8 numpy array. If Pillow is not
      installed, the image is returned unchanged and a message is
      logged: drawing needs Pillow.
    """
    img = _to_numpy_image(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not _PIL_AVAILABLE:
        logger.warning(
            "visualize_image: Pillow is not installed, returning the "
            "image unchanged (visualization needs Pillow).")
        return img
    if label_map is not None and not isinstance(label_map, dict):
        logger.warning(
            "visualize_image: label_map %r is not a dict; no label "
            "catalogue is bundled, so class ids are shown as "
            "'class_<id>'. Pass a dict {class_id: name} for real names."
            % (label_map,))

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    classes = np.asarray(classes).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    height, width = img.shape[0], img.shape[1]
    drawn = 0
    for idx in order:
        if drawn >= max_boxes_to_draw:
            break
        if scores[idx] < min_score_thresh:
            continue
        ymin, xmin, ymax, xmax = boxes[idx]
        x0 = int(max(0, min(width - 1, round(xmin))))
        y0 = int(max(0, min(height - 1, round(ymin))))
        x1 = int(max(0, min(width - 1, round(xmax))))
        y1 = int(max(0, min(height - 1, round(ymax))))
        for t in range(max(1, int(line_thickness))):
            draw.rectangle([x0 + t, y0 + t, x1 - t, y1 - t], outline=(255, 0, 0))
        label = "%s: %.2f" % (_label_for(int(classes[idx]), label_map),
                              float(scores[idx]))
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=(255, 0, 0))
        drawn += 1
    return np.array(pil_img)


def visualize_image_prediction(image, prediction, label_map=None, **kwargs):
    """Draw detection rows on an image.

    Args:
      image: image content with shape [height, width, 3].
      prediction: detections for one image, each row
        [image_id, ymin, xmin, ymax, xmax, score, class]. Rows for
        other image ids are ignored. May be an empty (0, 7) array.
      label_map: a map from label id to name (dict), or None.
      **kwargs: extra parameters for visualization, such as
        min_score_thresh, max_boxes_to_draw, and line_thickness.

    Returns:
      The annotated image as a uint8 numpy array.
    """
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 7)
    if prediction.shape[0] == 0:
        return visualize_image(image, np.zeros((0, 4)), np.zeros((0,), dtype=int),
                               np.zeros((0,)), label_map, **kwargs)
    rows = prediction[prediction[:, 0] == 0]
    if rows.shape[0] == 0:
        rows = prediction  # No image_id 0 rows: draw whatever was given.
    boxes = rows[:, 1:5]
    scores = rows[:, 5]
    classes = rows[:, 6].astype(int)
    return visualize_image(image, boxes, classes, scores, label_map, **kwargs)


# ---------------------------------------------------------------------------
# Serving driver
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "image_size": 512,
    "mean_rgb": (0.0, 0.0, 0.0),
    "stddev_rgb": (1.0, 1.0, 1.0),
    "label_map": None,
    "min_score_thresh": 0.05,
    "max_detections": 100,
    "nms_iou_thresh": 0.5,
}


class ServingDriver(object):
    """A driver for serving single or batch images.

    No model backend is bundled: without a `model_fn` every model path
    raises NoDetectionBackendError with a clear message. With a
    `model_fn` the full pipeline runs for real (preprocess -> model ->
    NMS post-processing -> visualization).

    A `model_fn` must accept ``model_fn(inputs, model_name=..., **params)``
    and return ``(cls_outputs, box_outputs)`` in the formats accepted
    by det_post_process (plain arrays or dicts keyed by feature level).

    Example 1 -- serving with a plugged-in backend:

      def my_model_fn(inputs, model_name=None, **params):
          ...  # run your model; return (cls_outputs, box_outputs)
          return cls_outputs, box_outputs

      driver = ServingDriver('my-model', '/tmp/my-model', batch_size=1,
                             model_fn=my_model_fn)
      driver.build()
      predictions = driver.serve_images([np.array(Image.open('a.jpg'))])
      driver.visualize(imgs[0], predictions)

    Example 2 -- serving from files:

      driver = ServingDriver('my-model', '/tmp/my-model')
      predictions = driver.serve_files(['/tmp/1.jpg', '/tmp/2.jpg'],
                                       model_fn=my_model_fn)

    Example 3 -- benchmarking a plugged-in backend:

      stats = driver.benchmark([img1, img2], num_runs=10,
                               model_fn=my_model_fn)
    """

    def __init__(self,
                 model_name,
                 ckpt_path,
                 batch_size=1,
                 use_xla=False,
                 min_score_thresh=None,
                 max_boxes_to_draw=None,
                 line_thickness=None,
                 model_params=None,
                 model_fn=None,
                 loader=None):
        """Initialize the inference driver.

        Args:
          model_name: target model name (kept for API compatibility;
            passed through to `model_fn`).
          ckpt_path: checkpoint path, such as /tmp/my-model/. Used by
            load()/restore_ckpt; no checkpoint is read without a
            `loader`.
          batch_size: batch size for inference.
          use_xla: kept for API compatibility; has no effect without a
            backend and is ignored.
          min_score_thresh: minimal score threshold for filtering
            predictions (default 0.05).
          max_boxes_to_draw: maximum number of boxes per image when
            visualizing (default 1000).
          line_thickness: line thickness for drawing boxes (default 2).
          model_params: model parameters overriding the defaults
            (image_size, mean_rgb, stddev_rgb, label_map,
            min_score_thresh, max_detections, nms_iou_thresh).
          model_fn: optional backend callable, see the class docstring.
          loader: optional checkpoint loader callable, see
            restore_ckpt.
        """
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.batch_size = int(batch_size)
        self.use_xla = use_xla
        self.model_fn = model_fn
        self.loader = loader

        self.params = dict(DEFAULT_PARAMS)
        if model_params:
            self.params.update(model_params)
        self.label_map = self.params.get("label_map")

        self.min_score_thresh = (min_score_thresh
                                 if min_score_thresh is not None
                                 else self.params.get("min_score_thresh", 0.05))
        self.max_boxes_to_draw = (max_boxes_to_draw
                                  if max_boxes_to_draw is not None else 1000)
        self.line_thickness = (line_thickness
                               if line_thickness is not None else 2)
        self.params["min_score_thresh"] = self.min_score_thresh

        self.signatures = None
        self._built = False

    # -- setup ----------------------------------------------------------

    def build(self, params_override=None, model_fn=None):
        """Wire in the model backend (or raise the honest error).

        Args:
          params_override: optional dict of params overriding
            self.params for this build.
          model_fn: optional backend callable; overrides the one passed
            to the constructor for this build.

        Returns:
          A dict describing the wired pipeline (keys: 'model_name',
          'model_fn', 'params', 'built').

        Raises:
          NoDetectionBackendError: if no `model_fn` is available.
        """
        if model_fn is not None:
            self.model_fn = model_fn
        params = dict(self.params)
        if params_override:
            params.update(params_override)
        self.params = params

        if self.model_fn is None:
            raise NoDetectionBackendError(
                "ServingDriver.build() has no model backend: this repo "
                "bundles no detection model, so there is nothing to "
                "build for model %r. Pass a model_fn callable -- "
                "model_fn(inputs, model_name=%r, **params) returning "
                "(cls_outputs, box_outputs) -- to the constructor or to "
                "build(model_fn=...). No model is faked here."
                % (self.model_name, self.model_name))
        self.signatures = {
            "model_name": self.model_name,
            "model_fn": self.model_fn,
            "params": dict(self.params),
            "built": True,
        }
        self._built = True
        return self.signatures

    def load(self, saved_model_dir_or_frozen_graph, loader=None):
        """Load a saved model or frozen graph via a supplied loader.

        Args:
          saved_model_dir_or_frozen_graph: path to the saved model.
          loader: optional callable ``loader(path, driver=self)`` that
            loads the model and returns a `model_fn`. Overrides the
            constructor's loader.

        Returns:
          The build signatures dict (see build()).

        Raises:
          FileNotFoundError: if the path does not exist.
          NoDetectionBackendError: if no loader is available.
        """
        if loader is not None:
            self.loader = loader
        if not os.path.exists(saved_model_dir_or_frozen_graph):
            raise FileNotFoundError(
                "saved model path does not exist: %s"
                % saved_model_dir_or_frozen_graph)
        if self.loader is None:
            raise NoDetectionBackendError(
                "ServingDriver.load(%r) has no loader: this repo bundles "
                "no saved-model format and no TensorFlow, so the path "
                "cannot be read. Pass a loader callable -- "
                "loader(path, driver=self) returning a model_fn -- to "
                "the constructor or to load(loader=...). No model is "
                "faked here." % saved_model_dir_or_frozen_graph)
        self.model_fn = self.loader(saved_model_dir_or_frozen_graph,
                                    driver=self)
        return self.build()

    def export(self, export_path, **kwargs):
        """Export the model -- not supported without a backend."""
        raise NoDetectionBackendError(
            "ServingDriver.export() has no model backend: there is no "
            "model in this repo to export. Wire in a model_fn first, "
            "then export through your own backend.")

    # -- serving ---------------------------------------------------------

    def _require_backend(self, model_fn=None):
        """Return a usable model_fn or raise the honest error."""
        if model_fn is not None:
            self.model_fn = model_fn
        if not self._built:
            self.build()
        if self.model_fn is None:
            raise NoDetectionBackendError(
                "No detection model backend is bundled with this repo: "
                "pass a model_fn callable to serve. No model is faked "
                "here.")
        return self.model_fn

    def _run(self, raw_images, model_fn=None):
        """Preprocess -> model_fn -> NMS; return the detections array."""
        model_fn = self._require_backend(model_fn)
        images, scales = batch_image_preprocess(
            raw_images, self.params["image_size"],
            self.params["mean_rgb"], self.params["stddev_rgb"])
        cls_outputs, box_outputs = model_fn(
            images, model_name=self.model_name, **self.params)
        return det_post_process(self.params, cls_outputs, box_outputs,
                                scales)

    def serve_files(self, image_files, model_fn=None):
        """Serve a list of input image files.

        Args:
          image_files: a list of image file paths.
          model_fn: optional backend callable (overrides the stored one).

        Returns:
          Detections as a (total, 7) array; each row is
          [image_id, ymin, xmin, ymax, xmax, score, class].
        """
        raw_images = batch_image_files_decode(image_files)
        return self._run(raw_images, model_fn=model_fn)

    def serve_images(self, image_arrays, model_fn=None):
        """Serve a list of image arrays.

        Args:
          image_arrays: a list of images (numpy arrays or PIL images).

        Returns:
          Detections as a (total, 7) array; each row is
          [image_id, ymin, xmin, ymax, xmax, score, class].
        """
        return self._run(list(image_arrays), model_fn=model_fn)

    def benchmark(self, image_arrays, num_runs=10, model_fn=None,
                  trace_filename=None):
        """Benchmark inference latency/throughput with a plugged-in backend.

        Args:
          image_arrays: a list of images in numpy array format.
          num_runs: number of timed runs (default 10).
          model_fn: optional backend callable.
          trace_filename: accepted for API compatibility; tracing is
            not supported without a backend framework, so it is ignored
            with a printed note.

        Returns:
          A dict with per-batch seconds, total seconds, number of runs,
          and images per second (FPS).
        """
        model_fn = self._require_backend(model_fn)
        if trace_filename:
            print("benchmark: trace_filename %r ignored -- tracing is "
                  "not supported without a backend framework."
                  % trace_filename)
        # Warm-up run (not timed).
        self._run(list(image_arrays), model_fn=model_fn)
        start = time.perf_counter()
        for _ in range(num_runs):
            self._run(list(image_arrays), model_fn=model_fn)
        end = time.perf_counter()
        total = end - start
        per_batch = total / max(1, num_runs)
        fps = self.batch_size / per_batch if per_batch > 0 else float("inf")
        print("Per batch inference time: %s" % per_batch)
        print("FPS: %s" % fps)
        return {
            "per_batch_seconds": per_batch,
            "total_seconds": total,
            "num_runs": num_runs,
            "batch_size": self.batch_size,
            "fps": fps,
        }

    def visualize(self, image, prediction, **kwargs):
        """Visualize predictions on an image."""
        return visualize_image_prediction(
            image,
            prediction,
            label_map=self.label_map,
            min_score_thresh=kwargs.pop("min_score_thresh",
                                        self.min_score_thresh),
            max_boxes_to_draw=kwargs.pop("max_boxes_to_draw",
                                         self.max_boxes_to_draw),
            line_thickness=kwargs.pop("line_thickness",
                                      self.line_thickness),
            **kwargs)


# ---------------------------------------------------------------------------
# Demo / smoke test (no model, no TensorFlow, no network)
# ---------------------------------------------------------------------------

def _demo():
    """Run the model-free smoke test and print the results.

    Builds a synthetic image, preprocesses it (single + batched),
    runs det_post_process on fake class/box outputs that include two
    heavily overlapping boxes of the same class (NMS must keep only
    the higher-scoring one), visualizes the result, and proves the
    honest no-backend error by calling ServingDriver.build() with no
    model_fn and catching it. Exits 0.
    """
    rng = np.random.default_rng(7)
    # Synthetic 240x320 RGB image: a gradient plus a bright square.
    yy, xx = np.mgrid[0:240, 0:320]
    image = np.stack([(xx % 256).astype(np.uint8),
                      (yy % 256).astype(np.uint8),
                      ((xx + yy) % 256).astype(np.uint8)], axis=-1)
    image[60:140, 100:220] = 200
    print("synthetic image shape: %s dtype: %s" % (image.shape, image.dtype))

    processed, scale = image_preprocess(image, 128, 0.0, 1.0)
    print("image_preprocess -> shape %s scale %s"
          % (processed.shape, scale))
    assert processed.shape == (128, 128, 3)
    assert scale == (240 / 128.0, 320 / 128.0)

    batch, scales = batch_image_preprocess([image, image], 128, 0.0, 1.0)
    print("batch_image_preprocess -> shape %s scales %s"
          % (batch.shape, scales))
    assert batch.shape == (2, 128, 128, 3)
    assert len(scales) == 2

    # Fake predictions for ONE image, 3 classes, 4 anchors.
    # Anchor 0 and anchor 1 are the same class, heavily overlapping
    # (IoU 0.87): NMS must keep anchor 0 (score 0.95) and drop 1.
    # Anchor 2 is a different class at the same place: it survives.
    # Anchor 3 scores below threshold: dropped.
    cls = np.array([[0.95, 0.02, 0.01],
                    [0.80, 0.02, 0.01],
                    [0.05, 0.90, 0.01],
                    [0.01, 0.02, 0.03]], dtype=np.float64)
    boxes = np.array([[10.0, 20.0, 60.0, 90.0],
                      [12.0, 22.0, 58.0, 88.0],
                      [10.0, 20.0, 60.0, 90.0],
                      [100.0, 100.0, 120.0, 120.0]], dtype=np.float64)
    detections = det_post_process(
        {"min_score_thresh": 0.05, "nms_iou_thresh": 0.5,
         "max_detections": 100},
        cls, boxes, scales=[(1.0, 1.0)])
    print("det_post_process -> %d detections:" % len(detections))
    for row in detections:
        print("  image_id=%d ymin=%.1f xmin=%.1f ymax=%.1f xmax=%.1f "
              "score=%.2f class=%d" % tuple(row))
    assert len(detections) == 2, "NMS should keep 2 of 4 anchors"
    assert detections[0][5] == 0.95 and detections[0][6] == 0
    assert detections[1][5] == 0.90 and detections[1][6] == 1

    annotated = visualize_image_prediction(image, detections)
    print("visualize_image_prediction -> shape %s dtype %s"
          % (annotated.shape, annotated.dtype))
    assert annotated.shape == image.shape
    # Drawing added red pixels (the box outlines / labels).
    assert np.any(annotated[:, :, 0] > image[:, :, 0]), \
        "visualization should have drawn on the image"

    # The honest no-backend error: build() with no model_fn.
    driver = ServingDriver("demo-model", "/tmp/no-such-ckpt", batch_size=1)
    try:
        driver.build()
    except NoDetectionBackendError as exc:
        print("ServingDriver.build() with no model_fn -> "
              "NoDetectionBackendError (as designed):")
        print("  %s" % exc)
    else:
        raise AssertionError("build() with no model_fn should have raised")

    # And the same error via restore_ckpt on a path that exists.
    try:
        restore_ckpt(None, __file__)
    except NoDetectionBackendError as exc:
        print("restore_ckpt(existing path, no loader) -> "
              "NoDetectionBackendError (as designed): %s" % exc)
    else:
        raise AssertionError("restore_ckpt with no loader should raise")

    # A plugged-in fake backend DOES run end to end.
    def fake_model_fn(inputs, model_name=None, **params):
        batch = inputs.shape[0]
        fake_cls = np.tile(cls[None, :, :], (batch, 1, 1))
        fake_box = np.tile(boxes[None, :, :], (batch, 1, 1))
        return fake_cls, fake_box

    driver2 = ServingDriver("demo-model", "/tmp/no-such-ckpt", batch_size=2,
                            model_fn=fake_model_fn)
    driver2.build()
    served = driver2.serve_images([image, image])
    print("ServingDriver with fake model_fn -> %d detections over 2 images"
          % len(served))
    assert len(served) == 4
    stats = driver2.benchmark([image, image], num_runs=3)
    print("benchmark -> %.4f s/batch, %.1f FPS"
          % (stats["per_batch_seconds"], stats["fps"]))

    print("demo: OK -- all checks passed (no model, no TF, no network)")
    return 0


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        raise SystemExit(_demo())
    print(__doc__)
    print("Usage: python nexus_agi_ultimate_system.py --demo")
    raise SystemExit(0)
