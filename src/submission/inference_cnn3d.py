from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter1d, zoom


OUTPUT_SHAPE_XYZ = np.asarray(
    [64, 64, 48],
    dtype=int
)

OUTPUT_SPACING_MM = 2.46

FOV_MM = (
    OUTPUT_SHAPE_XYZ.astype(np.float32)
    * OUTPUT_SPACING_MM
)


def _valid_groups(channels, maximum=8):
    for groups in (8, 4, 2, 1):
        if groups <= maximum and channels % groups == 0:
            return groups

    return 1


class ResidualBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout=0.0
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.norm1 = nn.GroupNorm(
            _valid_groups(out_channels),
            out_channels
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(
            _valid_groups(out_channels),
            out_channels
        )

        self.dropout = nn.Dropout3d(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.GroupNorm(
                    _valid_groups(out_channels),
                    out_channels
                )
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs):
        identity = self.shortcut(inputs)

        outputs = self.conv1(inputs)
        outputs = self.norm1(outputs)
        outputs = F.silu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm2(outputs)
        outputs = outputs + identity

        return F.silu(outputs)


class CompactCNN3D(nn.Module):

    def __init__(self, in_channels=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                16,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False
            ),
            nn.GroupNorm(4, 16),
            nn.SiLU()
        )

        self.features = nn.Sequential(
            ResidualBlock3D(16, 16, stride=1, dropout=0.05),
            ResidualBlock3D(16, 32, stride=2, dropout=0.05),
            ResidualBlock3D(32, 64, stride=2, dropout=0.10),
            ResidualBlock3D(64, 96, stride=2, dropout=0.10),
            ResidualBlock3D(96, 128, stride=2, dropout=0.15)
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 96),
            nn.SiLU(),
            nn.Dropout(0.40),
            nn.Linear(96, 1)
        )

    def forward(self, inputs):
        features = self.stem(inputs)
        features = self.features(features)

        average = self.avg_pool(features).flatten(1)
        maximum = self.max_pool(features).flatten(1)

        pooled = torch.cat(
            [average, maximum],
            dim=1
        )

        return self.classifier(pooled).squeeze(1)


def _robust_normalization(volume):
    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    volume = np.maximum(volume, 0.0)
    positive = volume[volume > 0.0]

    if positive.size == 0:
        raise ValueError(
            "El volumen no contiene intensidades positivas."
        )

    lower = float(np.percentile(positive, 1.0))
    upper = float(np.percentile(positive, 99.8))

    if upper <= lower:
        raise ValueError(
            "El rango de intensidades del volumen no es válido."
        )

    normalized = (volume - lower) / (upper - lower)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(
        np.float32,
        copy=False
    )


def _central_interval(length, fraction):
    margin = int(
        round(length * (1.0 - fraction) / 2.0)
    )

    return margin, length - margin


def _estimate_uptake_center(volume, spacing):
    nx, ny, nz = volume.shape

    x0, x1 = _central_interval(nx, 0.70)
    y0, y1 = _central_interval(ny, 0.70)
    z0, z1 = _central_interval(nz, 0.90)

    region = volume[x0:x1, y0:y1, z0:z1]
    positive = region[region > 0.0]

    if positive.size == 0:
        return np.asarray(
            [
                (nx - 1) / 2.0,
                (ny - 1) / 2.0,
                (nz - 1) / 2.0
            ],
            dtype=np.float32
        )

    threshold = float(
        np.percentile(positive, 97.5)
    )

    excess = np.maximum(region - threshold, 0.0)
    axial_profile = excess.sum(axis=(0, 1))

    sigma_z = max(
        1.0,
        5.0 / float(spacing[2])
    )

    axial_profile = gaussian_filter1d(
        axial_profile.astype(np.float32),
        sigma=sigma_z
    )

    center_z = z0 + int(np.argmax(axial_profile))

    radius_z = max(
        2,
        int(round(25.0 / float(spacing[2])))
    )

    za = max(0, center_z - radius_z)
    zb = min(nz, center_z + radius_z + 1)

    band = volume[x0:x1, y0:y1, za:zb]
    band_positive = band[band > 0.0]

    if band_positive.size == 0:
        center_x = (nx - 1) / 2.0
        center_y = (ny - 1) / 2.0
    else:
        xy_threshold = float(
            np.percentile(band_positive, 97.5)
        )

        weights = np.maximum(
            band - xy_threshold,
            0.0
        ) ** 2

        total_weight = float(weights.sum())

        if total_weight > 0.0:
            coordinates = np.indices(
                weights.shape,
                dtype=np.float32
            )

            center_x = (
                x0
                + float((coordinates[0] * weights).sum())
                / total_weight
            )

            center_y = (
                y0
                + float((coordinates[1] * weights).sum())
                / total_weight
            )
        else:
            center_x = (nx - 1) / 2.0
            center_y = (ny - 1) / 2.0

    return np.asarray(
        [center_x, center_y, center_z],
        dtype=np.float32
    )


def _extract_with_padding(volume, start, stop):
    start = np.asarray(start, dtype=int)
    stop = np.asarray(stop, dtype=int)
    output_shape = stop - start

    output = np.zeros(
        tuple(output_shape),
        dtype=np.float32
    )

    source_start = np.maximum(start, 0)
    source_stop = np.minimum(
        stop,
        np.asarray(volume.shape)
    )

    destination_start = source_start - start
    destination_stop = (
        destination_start
        + source_stop
        - source_start
    )

    if np.all(source_stop > source_start):
        output[
            destination_start[0]:destination_stop[0],
            destination_start[1]:destination_stop[1],
            destination_start[2]:destination_stop[2]
        ] = volume[
            source_start[0]:source_stop[0],
            source_start[1]:source_stop[1],
            source_start[2]:source_stop[2]
        ]

    return output


def preprocess_nifti_for_cnn3d(nifti_path):
    image = nib.as_closest_canonical(
        nib.load(str(nifti_path))
    )

    volume = image.get_fdata(dtype=np.float32)
    spacing = np.asarray(
        image.header.get_zooms()[:3],
        dtype=np.float32
    )

    volume = _robust_normalization(volume)
    center = _estimate_uptake_center(volume, spacing)

    crop_shape = np.maximum(
        3,
        np.round(FOV_MM / spacing).astype(int)
    )

    crop_shape += (crop_shape + 1) % 2

    start = np.round(
        center - crop_shape / 2.0
    ).astype(int)

    stop = start + crop_shape

    cropped = _extract_with_padding(
        volume,
        start,
        stop
    )

    zoom_factors = (
        OUTPUT_SHAPE_XYZ
        / np.asarray(cropped.shape)
    )

    resized = zoom(
        cropped,
        zoom=zoom_factors,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False
    )

    output = np.zeros(
        tuple(OUTPUT_SHAPE_XYZ),
        dtype=np.float32
    )

    limits = np.minimum(
        np.asarray(resized.shape),
        OUTPUT_SHAPE_XYZ
    )

    output[
        :limits[0],
        :limits[1],
        :limits[2]
    ] = resized[
        :limits[0],
        :limits[1],
        :limits[2]
    ]

    return np.clip(
        output,
        0.0,
        1.0
    ).astype(np.float32)


def _load_checkpoint(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


def load_cnn3d_models(
    stratified_paths,
    grouped_paths,
    device=None
):
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def load_family(paths):
        models = []

        for raw_path in paths:
            path = Path(raw_path)

            if not path.is_file():
                raise FileNotFoundError(
                    f"No se encontró el modelo CNN 3D: {path}"
                )

            checkpoint = _load_checkpoint(path)
            state_dict = checkpoint["model_state_dict"]

            model = CompactCNN3D(in_channels=2)
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            model.eval()
            models.append(model)

        return models

    stratified_models = load_family(stratified_paths)
    grouped_models = load_family(grouped_paths)

    return {
        "stratified": stratified_models,
        "grouped": grouped_models,
        "device": device
    }


def _safe_logit(probability, clip_value=1e-6):
    probability = float(
        np.clip(
            probability,
            clip_value,
            1.0 - clip_value
        )
    )

    return float(
        np.log(
            probability / (1.0 - probability)
        )
    )


def predict_cnn3d_hybrid_probability(
    nifti_path,
    model_bundle
):
    volume_xyz = preprocess_nifti_for_cnn3d(
        nifti_path
    )

    volume_zyx = np.transpose(
        volume_xyz,
        (2, 1, 0)
    )

    reflected = volume_zyx[:, :, ::-1].copy()
    asymmetry = volume_zyx - reflected

    model_input = np.stack(
        [volume_zyx, asymmetry],
        axis=0
    ).astype(np.float32)

    input_tensor = torch.from_numpy(
        np.ascontiguousarray(model_input)
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        model_bundle["device"]
    )

    def predict_family(models):
        probabilities = []

        with torch.inference_mode():
            for model in models:
                probability = torch.sigmoid(
                    model(input_tensor)
                )[0]

                probabilities.append(
                    float(probability.detach().cpu())
                )

        return float(np.mean(probabilities))

    stratified_probability = predict_family(
        model_bundle["stratified"]
    )

    grouped_probability = predict_family(
        model_bundle["grouped"]
    )

    hybrid_logit = (
        0.50 * _safe_logit(stratified_probability)
        + 0.50 * _safe_logit(grouped_probability)
    )

    hybrid_probability = float(
        1.0 / (1.0 + np.exp(-hybrid_logit))
    )

    diagnostics = {
        "stratified_probability": stratified_probability,
        "grouped_probability": grouped_probability,
        "hybrid_logit": hybrid_logit
    }

    return hybrid_probability, diagnostics
