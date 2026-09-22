import numpy as np

from scipy.special import expit


EPSILON = 1e-6

ROI_CONFIG = {
    "wide": {
        "x": (18, 78),
        "y": (18, 78),
        "z": (26, 39)
    },
    "narrow": {
        "x": (24, 72),
        "y": (20, 76),
        "z": (26, 39)
    }
}


def safe_correlation(a, b):

    a = np.asarray(
        a,
        dtype=np.float64
    ).reshape(-1)

    b = np.asarray(
        b,
        dtype=np.float64
    ).reshape(-1)

    if (
        a.size == 0 or
        b.size == 0 or
        np.std(a) < EPSILON or
        np.std(b) < EPSILON
    ):
        return 0.0

    correlation = np.corrcoef(
        a,
        b
    )[0, 1]

    if not np.isfinite(correlation):
        return 0.0

    return float(correlation)


def normalized_difference(left, right):

    return float(
        (left - right) /
        (abs(left) + abs(right) + EPSILON)
    )


def absolute_asymmetry(left, right):

    return float(
        abs(left - right) /
        (abs(left) + abs(right) + EPSILON)
    )


def weighted_center_of_mass(array):

    weights = np.asarray(
        array,
        dtype=np.float64
    )

    total = weights.sum()

    if total <= EPSILON:
        return (
            0.5,
            0.5,
            0.5
        )

    coordinates = np.indices(
        weights.shape,
        dtype=np.float64
    )

    centers = [
        float(
            (coordinates[axis] * weights).sum() /
            total
        )
        for axis in range(3)
    ]

    normalized_centers = []

    for center, size in zip(
        centers,
        weights.shape
    ):

        denominator = max(
            size - 1,
            1
        )

        normalized_centers.append(
            center / denominator
        )

    return tuple(
        normalized_centers
    )


def side_statistics(side, threshold):

    side = np.asarray(
        side,
        dtype=np.float64
    )

    positive = side[
        side > 0
    ]

    if positive.size == 0:
        positive = np.array(
            [0.0],
            dtype=np.float64
        )

    high_mask = (
        side >= threshold
    )

    high_values = side[
        high_mask
    ]

    return {
        "mean": float(side.mean()),
        "std": float(side.std()),
        "p50": float(np.percentile(positive, 50)),
        "p75": float(np.percentile(positive, 75)),
        "p90": float(np.percentile(positive, 90)),
        "p95": float(np.percentile(positive, 95)),
        "p99": float(np.percentile(positive, 99)),
        "sum": float(side.sum()),
        "nonzero_fraction": float(
            np.mean(side > 0)
        ),
        "high_fraction": float(
            high_mask.mean()
        ),
        "high_sum": float(
            high_values.sum()
            if high_values.size > 0
            else 0.0
        )
    }


def extract_features_from_single_roi(
    roi,
    prefix
):

    roi = np.asarray(
        roi,
        dtype=np.float32
    )

    if roi.shape[0] % 2 != 0:
        raise ValueError(
            f"La ROI {prefix} debe tener tamaño X par."
        )

    midpoint = (
        roi.shape[0] // 2
    )

    # En RAS, el eje 0 corresponde al eje izquierda-derecha.
    left = roi[
        :midpoint,
        :,
        :
    ]

    right = roi[
        midpoint:,
        :,
        :
    ]

    # Reflejar el hemisferio derecho para compararlo con el izquierdo.
    right_mirrored = right[
        ::-1,
        :,
        :
    ]

    positive = roi[
        roi > 0
    ]

    if positive.size == 0:
        threshold = 0.0
    else:
        threshold = float(
            np.percentile(
                positive,
                85
            )
        )

    left_stats = side_statistics(
        left,
        threshold
    )

    right_stats = side_statistics(
        right,
        threshold
    )

    features = {}

    # --------------------------------------------------------
    # Estadísticas globales de la ROI
    # --------------------------------------------------------

    features[f"{prefix}_global_mean"] = float(
        roi.mean()
    )

    features[f"{prefix}_global_std"] = float(
        roi.std()
    )

    features[f"{prefix}_global_p75"] = float(
        np.percentile(roi, 75)
    )

    features[f"{prefix}_global_p90"] = float(
        np.percentile(roi, 90)
    )

    features[f"{prefix}_global_p95"] = float(
        np.percentile(roi, 95)
    )

    features[f"{prefix}_global_p99"] = float(
        np.percentile(roi, 99)
    )

    features[f"{prefix}_global_sum"] = float(
        roi.sum()
    )

    features[f"{prefix}_global_nonzero_fraction"] = float(
        np.mean(roi > 0)
    )

    features[f"{prefix}_threshold_p85_positive"] = (
        threshold
    )

    # --------------------------------------------------------
    # Estadísticas de cada hemisferio
    # --------------------------------------------------------

    for statistic_name in left_stats:

        features[
            f"{prefix}_left_{statistic_name}"
        ] = left_stats[statistic_name]

        features[
            f"{prefix}_right_{statistic_name}"
        ] = right_stats[statistic_name]

    # --------------------------------------------------------
    # Asimetrías por estadística
    # --------------------------------------------------------

    asymmetry_statistics = [
        "mean",
        "std",
        "p75",
        "p90",
        "p95",
        "p99",
        "sum",
        "nonzero_fraction",
        "high_fraction",
        "high_sum"
    ]

    for statistic_name in asymmetry_statistics:

        left_value = left_stats[
            statistic_name
        ]

        right_value = right_stats[
            statistic_name
        ]

        features[
            f"{prefix}_{statistic_name}_signed_asymmetry"
        ] = normalized_difference(
            left_value,
            right_value
        )

        features[
            f"{prefix}_{statistic_name}_absolute_asymmetry"
        ] = absolute_asymmetry(
            left_value,
            right_value
        )

    # --------------------------------------------------------
    # Comparación vóxel a vóxel con reflexión
    # --------------------------------------------------------

    difference = (
        left.astype(np.float64) -
        right_mirrored.astype(np.float64)
    )

    features[
        f"{prefix}_mirror_correlation"
    ] = safe_correlation(
        left,
        right_mirrored
    )

    features[
        f"{prefix}_mirror_mae"
    ] = float(
        np.mean(
            np.abs(difference)
        )
    )

    features[
        f"{prefix}_mirror_rmse"
    ] = float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )

    dot_product = float(
        np.sum(
            left.astype(np.float64) *
            right_mirrored.astype(np.float64)
        )
    )

    left_norm = float(
        np.sqrt(
            np.sum(
                left.astype(np.float64) ** 2
            )
        )
    )

    right_norm = float(
        np.sqrt(
            np.sum(
                right_mirrored.astype(np.float64) ** 2
            )
        )
    )

    features[
        f"{prefix}_mirror_cosine_similarity"
    ] = (
        dot_product /
        (
            left_norm *
            right_norm +
            EPSILON
        )
    )

    # --------------------------------------------------------
    # Centros de masa por hemisferio
    # --------------------------------------------------------

    left_com = weighted_center_of_mass(
        left
    )

    right_com = weighted_center_of_mass(
        right_mirrored
    )

    axis_names = [
        "x",
        "y",
        "z"
    ]

    for axis_name, left_value, right_value in zip(
        axis_names,
        left_com,
        right_com
    ):

        features[
            f"{prefix}_left_com_{axis_name}"
        ] = float(left_value)

        features[
            f"{prefix}_right_com_{axis_name}"
        ] = float(right_value)

        features[
            f"{prefix}_com_{axis_name}_absolute_difference"
        ] = float(
            abs(
                left_value -
                right_value
            )
        )

    # --------------------------------------------------------
    # Asimetría a través de los cortes axiales
    # --------------------------------------------------------

    left_z_profile = left.sum(
        axis=(0, 1)
    )

    right_z_profile = right.sum(
        axis=(0, 1)
    )

    z_asymmetry = (
        np.abs(
            left_z_profile -
            right_z_profile
        ) /
        (
            np.abs(left_z_profile) +
            np.abs(right_z_profile) +
            EPSILON
        )
    )

    features[
        f"{prefix}_z_asymmetry_mean"
    ] = float(
        z_asymmetry.mean()
    )

    features[
        f"{prefix}_z_asymmetry_std"
    ] = float(
        z_asymmetry.std()
    )

    features[
        f"{prefix}_z_asymmetry_max"
    ] = float(
        z_asymmetry.max()
    )

    features[
        f"{prefix}_z_asymmetry_p90"
    ] = float(
        np.percentile(
            z_asymmetry,
            90
        )
    )

    return features


def extract_roi_asymmetry_features(volume):

    if volume.shape != (96, 96, 64):
        raise ValueError(
            f"Forma inesperada: {volume.shape}"
        )

    all_features = {}

    for roi_name, config in ROI_CONFIG.items():

        x0, x1 = config["x"]
        y0, y1 = config["y"]
        z0, z1 = config["z"]

        roi = volume[
            x0:x1,
            y0:y1,
            z0:z1
        ]

        roi_features = (
            extract_features_from_single_roi(
                roi,
                prefix=roi_name
            )
        )

        all_features.update(
            roi_features
        )

    feature_names = list(
        all_features.keys()
    )

    feature_values = np.asarray(
        [
            all_features[name]
            for name in feature_names
        ],
        dtype=np.float32
    )

    feature_values = np.nan_to_num(
        feature_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        feature_values,
        feature_names
    )


def predict_roi_probability(
    volume,
    parameters
):
    """
    Extrae las 136 variables ROI y calcula la probabilidad
    utilizando únicamente NumPy.
    """

    features, feature_names = (
        extract_roi_asymmetry_features(
            volume
        )
    )

    saved_feature_names = (
        parameters[
            "feature_names"
        ].astype(str)
    )

    current_feature_names = np.asarray(
        feature_names,
        dtype=str
    )

    if not np.array_equal(
        current_feature_names,
        saved_feature_names
    ):
        raise ValueError(
            "El orden de las características ROI "
            "no coincide con el modelo guardado."
        )

    scaler_mean = parameters[
        "scaler_mean"
    ].astype(np.float64)

    scaler_scale = parameters[
        "scaler_scale"
    ].astype(np.float64)

    coefficients = parameters[
        "coefficients"
    ].astype(np.float64)

    intercept = float(
        parameters[
            "intercept"
        ][0]
    )

    features_scaled = (
        features.astype(np.float64) -
        scaler_mean
    ) / scaler_scale

    prediction_logit = float(
        features_scaled @ coefficients +
        intercept
    )

    probability = float(
        expit(
            prediction_logit
        )
    )

    if not np.isfinite(probability):
        raise ValueError(
            "La probabilidad ROI no es finita."
        )

    probability = float(
        np.clip(
            probability,
            1e-6,
            1 - 1e-6
        )
    )

    return probability, features
