
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm


IMAGENET_MEAN = np.asarray(
    [
        0.485,
        0.456,
        0.406
    ],
    dtype=np.float32
).reshape(
    1,
    3,
    1,
    1
)


IMAGENET_STD = np.asarray(
    [
        0.229,
        0.224,
        0.225
    ],
    dtype=np.float32
).reshape(
    1,
    3,
    1,
    1
)


def pad_to_square(
    image
):
    """
    Rellena una imagen rectangular con ceros para
    hacerla cuadrada sin deformar la anatomía.
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    height, width = image.shape

    target_size = max(
        height,
        width
    )

    pad_height = (
        target_size - height
    )

    pad_width = (
        target_size - width
    )

    pad_top = (
        pad_height // 2
    )

    pad_bottom = (
        pad_height - pad_top
    )

    pad_left = (
        pad_width // 2
    )

    pad_right = (
        pad_width - pad_left
    )

    padded = np.pad(
        image,
        (
            (
                pad_top,
                pad_bottom
            ),
            (
                pad_left,
                pad_right
            )
        ),
        mode="constant",
        constant_values=0.0
    )

    return padded

def resize_image(
    image,
    output_size=160
):
    """
    Redimensiona mediante interpolación bilineal.
    """

    square_image = pad_to_square(
        image
    )

    tensor = torch.from_numpy(
        square_image
    ).float()[
        None,
        None,
        :,
        :
    ]

    resized = F.interpolate(
        tensor,
        size=(
            output_size,
            output_size
        ),
        mode="bilinear",
        align_corners=False
    )

    return (
        resized[
            0,
            0
        ]
        .numpy()
        .astype(
            np.float32
        )
    )

def robust_channel_normalization_v2(
    image
):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    positive_values = image[
        image > 0.0
    ]

    if len(
        positive_values
    ) == 0:

        return np.zeros_like(
            image,
            dtype=np.float32
        )

    lower = np.percentile(
        positive_values,
        0.5
    )

    upper = np.percentile(
        positive_values,
        99.9
    )

    if upper <= lower:

        return np.zeros_like(
            image,
            dtype=np.float32
        )

    normalized = (
        image - lower
    ) / (
        upper - lower
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    ).astype(
        np.float32
    )

def extract_multiview_tensor_v2(
    centered_volume,
    output_size=160
):
    """
    Tensor multivista enfocado en la región estriatal.

    Salida:
        (3 vistas, 3 canales, 160, 160)
    """

    volume = np.asarray(
        centered_volume,
        dtype=np.float32
    )

    if volume.shape != (
        96,
        96,
        64
    ):

        raise ValueError(
            "El volumen debe tener forma (96, 96, 64). "
            f"Se recibió {volume.shape}."
        )

    if not np.isfinite(
        volume
    ).all():

        raise ValueError(
            "El volumen contiene valores no finitos."
        )


    # --------------------------------------------------------
    # CROP CENTRAL XY
    # --------------------------------------------------------

    center_x = (
        volume.shape[0] // 2
    )

    center_y = (
        volume.shape[1] // 2
    )

    crop_half_width = 32

    x_start = (
        center_x - crop_half_width
    )

    x_end = (
        center_x + crop_half_width
    )

    y_start = (
        center_y - crop_half_width
    )

    y_end = (
        center_y + crop_half_width
    )


    # --------------------------------------------------------
    # BUSCAR Z PEAK SOLAMENTE EN LA REGIÓN CENTRAL
    # --------------------------------------------------------

    peak_search_volume = volume[
        x_start:x_end,
        y_start:y_end,
        :
    ]

    axial_signal = np.sum(
        peak_search_volume,
        axis=(
            0,
            1
        )
    )

    z_peak = int(
        np.argmax(
            axial_signal
        )
    )


    # Limitar el pico a una zona anatómicamente razonable
    z_peak = int(
        np.clip(
            z_peak,
            12,
            volume.shape[2] - 13
        )
    )


    z_half_width = 12

    z_start = (
        z_peak - z_half_width
    )

    z_end = (
        z_peak + z_half_width + 1
    )


    # Crop estriatal 64 × 64 × 25
    striatal_crop = volume[
        x_start:x_end,
        y_start:y_end,
        z_start:z_end
    ]


    assert striatal_crop.shape == (
        64,
        64,
        25
    )


    crop_center_x = (
        striatal_crop.shape[0] // 2
    )

    crop_center_y = (
        striatal_crop.shape[1] // 2
    )

    crop_center_z = (
        striatal_crop.shape[2] // 2
    )


    # Slabs internos
    axial_slab_start = max(
        0,
        crop_center_z - 5
    )

    axial_slab_end = min(
        striatal_crop.shape[2],
        crop_center_z + 6
    )

    coronal_slab_start = max(
        0,
        crop_center_y - 10
    )

    coronal_slab_end = min(
        striatal_crop.shape[1],
        crop_center_y + 11
    )

    sagittal_slab_start = max(
        0,
        crop_center_x - 10
    )

    sagittal_slab_end = min(
        striatal_crop.shape[0],
        crop_center_x + 11
    )


    # --------------------------------------------------------
    # AXIAL: X × Y
    # --------------------------------------------------------

    axial_mip = np.max(
        striatal_crop,
        axis=2
    )

    axial_mean = np.mean(
        striatal_crop[
            :,
            :,
            axial_slab_start:
            axial_slab_end
        ],
        axis=2
    )

    axial_slice = striatal_crop[
        :,
        :,
        crop_center_z
    ]


    # --------------------------------------------------------
    # CORONAL: X × Z
    # --------------------------------------------------------

    coronal_mip = np.max(
        striatal_crop,
        axis=1
    )

    coronal_mean = np.mean(
        striatal_crop[
            :,
            coronal_slab_start:
            coronal_slab_end,
            :
        ],
        axis=1
    )

    coronal_slice = striatal_crop[
        :,
        crop_center_y,
        :
    ]


    # --------------------------------------------------------
    # SAGITAL: Y × Z
    # --------------------------------------------------------

    sagittal_mip = np.max(
        striatal_crop,
        axis=0
    )

    sagittal_mean = np.mean(
        striatal_crop[
            sagittal_slab_start:
            sagittal_slab_end,
            :,
            :
        ],
        axis=0
    )

    sagittal_slice = striatal_crop[
        crop_center_x,
        :,
        :
    ]


    raw_views = {
        "axial": [
            axial_mip,
            axial_mean,
            axial_slice
        ],
        "coronal": [
            coronal_mip,
            coronal_mean,
            coronal_slice
        ],
        "sagittal": [
            sagittal_mip,
            sagittal_mean,
            sagittal_slice
        ]
    }


    view_arrays = []

    for view_name in [
        "axial",
        "coronal",
        "sagittal"
    ]:

        channel_arrays = []

        for channel_image in raw_views[
            view_name
        ]:

            normalized = (
                robust_channel_normalization_v2(
                    channel_image
                )
            )

            resized = resize_image(
                normalized,
                output_size=output_size
            )

            channel_arrays.append(
                resized
            )

        view_arrays.append(
            np.stack(
                channel_arrays,
                axis=0
            )
        )


    multiview_tensor = np.stack(
        view_arrays,
        axis=0
    ).astype(
        np.float32
    )


    metadata = {
        "z_peak": z_peak,
        "crop_x": (
            x_start,
            x_end
        ),
        "crop_y": (
            y_start,
            y_end
        ),
        "crop_z": (
            z_start,
            z_end
        ),
        "crop_shape": (
            striatal_crop.shape
        )
    }


    return (
        multiview_tensor,
        metadata
    )


def load_efficientnet_multiview_model(
    encoder_path,
    linear_parameters_path
):
    """
    Carga EfficientNet-B0 y el modelo lineal portable.
    """

    encoder_path = Path(
        encoder_path
    )

    linear_parameters_path = Path(
        linear_parameters_path
    )

    if not encoder_path.exists():

        raise FileNotFoundError(
            f"No se encontró el encoder: {encoder_path}"
        )

    if not linear_parameters_path.exists():

        raise FileNotFoundError(
            "No se encontraron los parámetros lineales: "
            f"{linear_parameters_path}"
        )

    encoder = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=0,
        global_pool="avg"
    )

    try:

        encoder_state_dict = torch.load(
            encoder_path,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:

        encoder_state_dict = torch.load(
            encoder_path,
            map_location="cpu"
        )

    encoder.load_state_dict(
        encoder_state_dict,
        strict=True
    )

    encoder = encoder.cpu()
    encoder.eval()

    with np.load(
        linear_parameters_path,
        allow_pickle=False
    ) as saved_parameters:

        linear_parameters = {
            name: saved_parameters[
                name
            ].copy()
            for name in saved_parameters.files
        }

    expected_keys = {
        "scaler_mean",
        "scaler_scale",
        "coefficients",
        "intercept",
        "embedding_dimension",
        "view_embedding_dimension",
        "n_views",
        "current_weight",
        "cnn_weight"
    }

    missing_keys = (
        expected_keys
        -
        set(
            linear_parameters
        )
    )

    if missing_keys:

        raise KeyError(
            "Faltan parámetros del modelo lineal: "
            f"{sorted(missing_keys)}"
        )

    if int(
        linear_parameters[
            "embedding_dimension"
        ]
    ) != 3840:

        raise ValueError(
            "La dimensión esperada debe ser 3840."
        )

    return (
        encoder,
        linear_parameters
    )


def predict_efficientnet_probability(
    centered_volume,
    encoder,
    linear_parameters
):
    """
    Genera la probabilidad EfficientNet a partir de un
    volumen centrado de forma (96, 96, 64).
    """

    centered_volume = np.asarray(
        centered_volume,
        dtype=np.float32
    )

    if centered_volume.shape != (
        96,
        96,
        64
    ):

        raise ValueError(
            "Forma inesperada del volumen centrado: "
            f"{centered_volume.shape}"
        )

    if not np.isfinite(
        centered_volume
    ).all():

        raise ValueError(
            "El volumen contiene valores no finitos."
        )

    multiview_array, view_metadata = (
        extract_multiview_tensor_v2(
            centered_volume,
            output_size=160
        )
    )

    multiview_array = np.asarray(
        multiview_array,
        dtype=np.float32
    )

    if multiview_array.shape != (
        3,
        3,
        160,
        160
    ):

        raise ValueError(
            "Forma inesperada del tensor multivista: "
            f"{multiview_array.shape}"
        )

    if not np.isfinite(
        multiview_array
    ).all():

        raise ValueError(
            "El tensor multivista contiene valores no finitos."
        )

    normalized_views = (
        (
            multiview_array
            -
            IMAGENET_MEAN
        )
        /
        IMAGENET_STD
    )

    input_tensor = torch.from_numpy(
        np.ascontiguousarray(
            normalized_views
        )
    ).float()

    with torch.inference_mode():

        embeddings_by_view = encoder(
            input_tensor
        )

    embeddings_by_view = (
        embeddings_by_view
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    if embeddings_by_view.shape != (
        3,
        1280
    ):

        raise ValueError(
            "Forma inesperada de los embeddings: "
            f"{embeddings_by_view.shape}"
        )

    combined_embedding = (
        embeddings_by_view.reshape(
            -1
        )
    )

    if combined_embedding.shape != (
        3840,
    ):

        raise ValueError(
            "El embedding concatenado no tiene "
            "3840 elementos."
        )

    scaler_mean = np.asarray(
        linear_parameters[
            "scaler_mean"
        ],
        dtype=np.float64
    )

    scaler_scale = np.asarray(
        linear_parameters[
            "scaler_scale"
        ],
        dtype=np.float64
    )

    coefficients = np.asarray(
        linear_parameters[
            "coefficients"
        ],
        dtype=np.float64
    )

    intercept = float(
        linear_parameters[
            "intercept"
        ]
    )

    scaled_embedding = (
        (
            combined_embedding
            -
            scaler_mean
        )
        /
        scaler_scale
    )

    logit = float(
        scaled_embedding
        @
        coefficients
        +
        intercept
    )

    logit = float(
        np.clip(
            logit,
            -30.0,
            30.0
        )
    )

    probability = float(
        1.0
        /
        (
            1.0
            +
            np.exp(
                -logit
            )
        )
    )

    if not np.isfinite(
        probability
    ):

        raise ValueError(
            "La probabilidad EfficientNet no es finita."
        )

    if not (
        0.0
        <=
        probability
        <=
        1.0
    ):

        raise ValueError(
            "La probabilidad EfficientNet está fuera de rango."
        )

    return (
        probability,
        combined_embedding.astype(
            np.float32
        ),
        view_metadata
    )
