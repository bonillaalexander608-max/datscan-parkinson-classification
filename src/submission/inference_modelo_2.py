import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage
from scipy.special import expit, logit
from skimage.transform import resize


TARGET_SPACING = np.array(
    [2.5, 2.5, 2.5],
    dtype=np.float32
)

TARGET_FOV_MM = np.array(
    [240.0, 240.0, 160.0],
    dtype=np.float32
)

TARGET_SHAPE = np.array(
    [96, 96, 64],
    dtype=int
)

XY_SEARCH_FRACTION = 0.60
MAX_XY_SHIFT_FRACTION = 0.18


def crop_or_pad_around_center(
    array,
    target_shape,
    center
):
    """
    Extrae un volumen con target_shape alrededor de center.
    Completa con ceros cuando el recorte sale del volumen.
    """

    target_shape = np.asarray(
        target_shape,
        dtype=int
    )

    center = np.asarray(
        center,
        dtype=float
    )

    output = np.zeros(
        tuple(target_shape),
        dtype=array.dtype
    )

    source_slices = []
    destination_slices = []

    for axis in range(3):

        start = int(
            round(
                center[axis] -
                target_shape[axis] / 2
            )
        )

        stop = (
            start +
            target_shape[axis]
        )

        source_start = max(
            start,
            0
        )

        source_stop = min(
            stop,
            array.shape[axis]
        )

        destination_start = max(
            -start,
            0
        )

        destination_stop = (
            destination_start +
            source_stop -
            source_start
        )

        source_slices.append(
            slice(
                source_start,
                source_stop
            )
        )

        destination_slices.append(
            slice(
                destination_start,
                destination_stop
            )
        )

    output[
        tuple(destination_slices)
    ] = array[
        tuple(source_slices)
    ]

    return output


def center_crop_or_pad(
    array,
    target_shape
):
    """
    Recorta o completa con ceros usando el centro geométrico.
    """

    target_shape = np.asarray(
        target_shape,
        dtype=int
    )

    center = (
        np.asarray(
            array.shape,
            dtype=float
        ) - 1
    ) / 2

    return crop_or_pad_around_center(
        array,
        target_shape,
        center
    )


def robust_normalize(
    volume,
    percentile=99.5
):
    """
    Normaliza el volumen utilizando el percentil 99.5
    de sus intensidades positivas.
    """

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    positive = volume[
        np.isfinite(volume) &
        (volume > 0)
    ]

    if positive.size == 0:
        raise ValueError(
            "El volumen no contiene intensidades positivas."
        )

    scale = float(
        np.percentile(
            positive,
            percentile
        )
    )

    scale = max(scale, 1e-6)

    normalized = np.clip(
        volume / scale,
        0.0,
        1.0
    )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return normalized.astype(
        np.float32,
        copy=False
    )


def find_striatal_z_baseline(volume):
    """
    Localizador Z utilizado en modelo_1.ipynb.

    Busca captación superior al percentil 98.5 dentro
    de la región central X/Y.
    """

    nx, ny, nz = volume.shape

    x0 = int(0.20 * nx)
    x1 = int(0.80 * nx)

    y0 = int(0.20 * ny)
    y1 = int(0.80 * ny)

    central = volume[
        x0:x1,
        y0:y1,
        :
    ]

    central_positive = central[
        central > 0
    ]

    if central_positive.size == 0:
        return int(nz // 2)

    threshold = np.percentile(
        central_positive,
        98.5
    )

    high_uptake = np.maximum(
        central - threshold,
        0
    )

    slice_score = high_uptake.sum(
        axis=(0, 1)
    )

    if np.all(slice_score == 0):
        return int(nz // 2)

    return int(
        np.argmax(slice_score)
    )


def find_restricted_xy_center(
    volume,
    z_center,
    search_fraction=XY_SEARCH_FRACTION,
    max_shift_fraction=MAX_XY_SHIFT_FRACTION,
    z_half_window=8,
    centroid_strength=0.50
):
    """
    Estima el centro X/Y de captación dentro de una región
    central restringida.

    Protecciones:
    1. Solo busca en la zona central.
    2. Usa una ventana alrededor del centro Z.
    3. Conserva únicamente captación intensa.
    4. Limita cuánto puede alejarse del centro geométrico.
    5. Aplica solo una fracción del desplazamiento encontrado,
       reduciendo el efecto de la asimetría patológica.
    """

    nx, ny, nz = volume.shape

    geometric_x = (nx - 1) / 2
    geometric_y = (ny - 1) / 2

    # ------------------------------------------
    # Límites de búsqueda X/Y
    # ------------------------------------------

    search_size_x = int(
        round(nx * search_fraction)
    )

    search_size_y = int(
        round(ny * search_fraction)
    )

    x0 = max(
        0,
        int(round(geometric_x - search_size_x / 2))
    )

    x1 = min(
        nx,
        x0 + search_size_x
    )

    y0 = max(
        0,
        int(round(geometric_y - search_size_y / 2))
    )

    y1 = min(
        ny,
        y0 + search_size_y
    )

    # ------------------------------------------
    # Ventana alrededor del centro Z
    # ------------------------------------------

    z0 = max(
        0,
        int(z_center - z_half_window)
    )

    z1 = min(
        nz,
        int(z_center + z_half_window + 1)
    )

    roi = volume[
        x0:x1,
        y0:y1,
        z0:z1
    ]

    positive = roi[roi > 0]

    if positive.size == 0:
        return (
            float(geometric_x),
            float(geometric_y)
        )

    # Captación intensa dentro de la ROI
    threshold = float(
        np.percentile(
            positive,
            85
        )
    )

    weights = np.where(
        roi >= threshold,
        roi,
        0.0
    ).astype(np.float64)

    # Reducir el dominio de puntos extremadamente intensos
    high_values = weights[
        weights > 0
    ]

    if high_values.size > 0:
        upper_limit = float(
            np.percentile(
                high_values,
                99
            )
        )

        weights = np.minimum(
            weights,
            upper_limit
        )

    total_weight = weights.sum()

    if total_weight <= 1e-8:
        return (
            float(geometric_x),
            float(geometric_y)
        )

    # ------------------------------------------
    # Centro de masa dentro de la ROI
    # ------------------------------------------

    coordinates = np.indices(
        weights.shape,
        dtype=np.float64
    )

    local_x = (
        coordinates[0] * weights
    ).sum() / total_weight

    local_y = (
        coordinates[1] * weights
    ).sum() / total_weight

    uptake_x = x0 + local_x
    uptake_y = y0 + local_y

    # ------------------------------------------
    # Desplazamiento amortiguado
    # ------------------------------------------

    estimated_x = (
        geometric_x +
        centroid_strength *
        (uptake_x - geometric_x)
    )

    estimated_y = (
        geometric_y +
        centroid_strength *
        (uptake_y - geometric_y)
    )

    # ------------------------------------------
    # Limitar desplazamientos máximos
    # ------------------------------------------

    max_shift_x = nx * max_shift_fraction
    max_shift_y = ny * max_shift_fraction

    restricted_x = np.clip(
        estimated_x,
        geometric_x - max_shift_x,
        geometric_x + max_shift_x
    )

    restricted_y = np.clip(
        estimated_y,
        geometric_y - max_shift_y,
        geometric_y + max_shift_y
    )

    return (
        float(restricted_x),
        float(restricted_y)
    )


def preprocess_volume_xy_variant(
    path,
    use_restricted_xy=True
):
    """
    Reproduce el preprocesamiento del baseline.

    La única diferencia opcional es:
    - False: centro geométrico X/Y.
    - True: centro X/Y restringido.

    El método Z permanece idéntico en ambos casos.
    """

    image = nib.load(
        str(path)
    )

    image = nib.as_closest_canonical(
        image
    )

    volume = np.asarray(
        image.dataobj,
        dtype=np.float32
    )

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    volume[volume < 0] = 0

    original_spacing = np.asarray(
        image.header.get_zooms()[:3],
        dtype=np.float32
    )

    # ------------------------------------------
    # MISMO CENTRO Z PARA AMBAS VARIANTES
    # ------------------------------------------

    z_peak_original = (
        find_striatal_z_baseline(
            volume
        )
    )

    geometric_x = (
        volume.shape[0] - 1
    ) / 2

    geometric_y = (
        volume.shape[1] - 1
    ) / 2

    # ------------------------------------------
    # ÚNICA DIFERENCIA: CENTRO X/Y
    # ------------------------------------------

    if use_restricted_xy:

        normalized_for_center = (
            robust_normalize(
                volume
            )
        )

        # Aproximadamente ±20 mm alrededor de Z
        z_half_window = max(
            2,
            int(round(
                20.0 /
                max(
                    original_spacing[2],
                    1e-6
                )
            ))
        )

        center_x, center_y = (
            find_restricted_xy_center(
                normalized_for_center,
                z_peak_original,
                search_fraction=0.60,
                max_shift_fraction=0.18,
                z_half_window=z_half_window,
                centroid_strength=0.50
            )
        )

    else:

        center_x = geometric_x
        center_y = geometric_y

    center_original = (
        center_x,
        center_y,
        float(z_peak_original)
    )

    # ------------------------------------------
    # RECORTE EN EL ESPACIO ORIGINAL
    # ------------------------------------------

    desired_original_shape = np.ceil(
        TARGET_FOV_MM /
        original_spacing
    ).astype(int)

    cropped = crop_or_pad_around_center(
        volume,
        target_shape=desired_original_shape,
        center=center_original
    )

    # ------------------------------------------
    # RESAMPLEO
    # ------------------------------------------

    zoom_factors = (
        original_spacing /
        TARGET_SPACING
    )

    resampled = ndimage.zoom(
        cropped,
        zoom=zoom_factors,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False
    )

    resampled = center_crop_or_pad(
        resampled,
        TARGET_SHAPE
    )

    resampled = robust_normalize(
        resampled
    )

    metadata = {
        "z_peak_original": int(
            z_peak_original
        ),

        "geometric_x": float(
            geometric_x
        ),

        "geometric_y": float(
            geometric_y
        ),

        "center_x": float(
            center_x
        ),

        "center_y": float(
            center_y
        ),

        "delta_x": float(
            center_x -
            geometric_x
        ),

        "delta_y": float(
            center_y -
            geometric_y
        ),

        "spacing_x": float(
            original_spacing[0]
        ),

        "spacing_y": float(
            original_spacing[1]
        ),

        "spacing_z": float(
            original_spacing[2]
        )
    }

    return (
        resampled.astype(
            np.float32,
            copy=False
        ),
        metadata
    )


def extract_features_from_volume(
    volume,
    image_size=32
):
    """
    Reproduce las características del baseline:

    - tres proyecciones axiales de 32x32;
    - siete resúmenes estadísticos;
    - asimetría izquierda-derecha.

    Total:
    3*32*32 + 8 = 3,080 características.
    """

    if volume.shape != (96, 96, 64):
        raise ValueError(
            f"Forma inesperada: {volume.shape}"
        )

    z_center = volume.shape[2] // 2

    windows = [
        (z_center - 6, z_center - 1),
        (z_center - 2, z_center + 3),
        (z_center + 2, z_center + 7)
    ]

    image_features = []

    for z0, z1 in windows:

        z0 = int(
            np.clip(
                z0,
                0,
                volume.shape[2] - 1
            )
        )

        z1 = int(
            np.clip(
                z1,
                z0 + 1,
                volume.shape[2]
            )
        )

        projection = volume[
            :,
            :,
            z0:z1
        ].max(axis=2)

        projection_small = resize(
            projection,
            (image_size, image_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

        image_features.append(
            projection_small.ravel()
        )

    central_roi = volume[
        18:78,
        18:78,
        z_center - 6:z_center + 7
    ]

    midpoint = (
        central_roi.shape[0] // 2
    )

    left_sum = float(
        central_roi[:midpoint].sum()
    )

    right_sum = float(
        central_roi[midpoint:].sum()
    )

    lr_asymmetry = (
        abs(left_sum - right_sum) /
        (left_sum + right_sum + 1e-6)
    )

    scalar_features = np.array(
        [
            central_roi.mean(),
            central_roi.std(),
            np.percentile(
                central_roi,
                50
            ),
            np.percentile(
                central_roi,
                75
            ),
            np.percentile(
                central_roi,
                90
            ),
            np.percentile(
                central_roi,
                95
            ),
            np.percentile(
                central_roi,
                99
            ),
            lr_asymmetry
        ],
        dtype=np.float32
    )

    features = np.concatenate(
        image_features +
        [scalar_features]
    )

    return features.astype(
        np.float32,
        copy=False
    )


def extract_multiview_projections(
    volume,
    image_size=32
):
    """
    Extrae nueve proyecciones 2.5D:

    - 3 axiales: proyección a través del eje Z.
    - 3 coronales: proyección a través del eje Y.
    - 3 sagitales: proyección a través del eje X.

    Cada proyección se redimensiona a 32x32.

    Parámetros
    ----------
    volume : ndarray
        Volumen estandarizado de forma (96, 96, 64).

    image_size : int
        Tamaño final de cada proyección.

    Retorna
    -------
    axial : ndarray
        Forma (3, 32, 32).

    coronal : ndarray
        Forma (3, 32, 32).

    sagittal : ndarray
        Forma (3, 32, 32).
    """

    volume = np.asarray(
        volume,
        dtype=np.float32
    )

    if volume.shape != (96, 96, 64):
        raise ValueError(
            f"Se esperaba un volumen (96, 96, 64), "
            f"pero se recibió {volume.shape}."
        )

    center_x = volume.shape[0] // 2  # 48
    center_y = volume.shape[1] // 2  # 48
    center_z = volume.shape[2] // 2  # 32

    # Ventanas equivalentes alrededor de cada centro
    offsets = [
        (-6, -1),
        (-2,  3),
        ( 2,  7)
    ]

    axial_images = []
    coronal_images = []
    sagittal_images = []

    for start_offset, stop_offset in offsets:

        # ------------------------------------------
        # AXIAL: proyectar varios cortes en Z
        # Resultado espacial: X por Y
        # ------------------------------------------

        z0 = int(
            np.clip(
                center_z + start_offset,
                0,
                volume.shape[2] - 1
            )
        )

        z1 = int(
            np.clip(
                center_z + stop_offset,
                z0 + 1,
                volume.shape[2]
            )
        )

        axial_projection = volume[
            :,
            :,
            z0:z1
        ].max(axis=2)

        axial_small = resize(
            axial_projection,
            (image_size, image_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

        axial_images.append(
            axial_small
        )

        # ------------------------------------------
        # CORONAL: proyectar varios cortes en Y
        # Resultado espacial: X por Z
        # ------------------------------------------

        y0 = int(
            np.clip(
                center_y + start_offset,
                0,
                volume.shape[1] - 1
            )
        )

        y1 = int(
            np.clip(
                center_y + stop_offset,
                y0 + 1,
                volume.shape[1]
            )
        )

        coronal_projection = volume[
            :,
            y0:y1,
            :
        ].max(axis=1)

        coronal_small = resize(
            coronal_projection,
            (image_size, image_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

        coronal_images.append(
            coronal_small
        )

        # ------------------------------------------
        # SAGITAL: proyectar varios cortes en X
        # Resultado espacial: Y por Z
        # ------------------------------------------

        x0 = int(
            np.clip(
                center_x + start_offset,
                0,
                volume.shape[0] - 1
            )
        )

        x1 = int(
            np.clip(
                center_x + stop_offset,
                x0 + 1,
                volume.shape[0]
            )
        )

        sagittal_projection = volume[
            x0:x1,
            :,
            :
        ].max(axis=0)

        sagittal_small = resize(
            sagittal_projection,
            (image_size, image_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

        sagittal_images.append(
            sagittal_small
        )

    axial = np.stack(
        axial_images
    ).astype(np.float32)

    coronal = np.stack(
        coronal_images
    ).astype(np.float32)

    sagittal = np.stack(
        sagittal_images
    ).astype(np.float32)

    if not (
        np.isfinite(axial).all()
        and np.isfinite(coronal).all()
        and np.isfinite(sagittal).all()
    ):
        raise ValueError(
            "Las proyecciones contienen NaN o infinitos."
        )

    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal
    }


def multiscale_fourier_silent(
    X_view,
    scales=(16, 24, 32)
):
    """
    Versión silenciosa de Fourier multiescala.
    Mantiene exactamente el mismo cálculo utilizado
    durante el entrenamiento.
    """

    X_view = np.asarray(
        X_view,
        dtype=np.float32
    )

    if (
        X_view.ndim != 2
        or
        X_view.shape[1] != 3072
    ):
        raise ValueError(
            "X_view debe tener forma (n, 3072)."
        )

    images_32 = X_view.reshape(
        len(X_view),
        3,
        32,
        32
    )

    fft_by_scale = {}

    for scale in scales:

        scale_blocks = []

        for channel in range(3):

            channel_images = (
                images_32[
                    :,
                    channel
                ]
            )

            if scale == 32:

                resized_images = (
                    channel_images
                )

            else:

                resized_images = np.stack([
                    resize(
                        image,
                        (scale, scale),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True
                    )
                    for image in (
                        channel_images
                    )
                ]).astype(
                    np.float32
                )

            fft_channel = np.fft.fft2(
                resized_images,
                axes=(1, 2)
            )

            fft_channel = np.fft.fftshift(
                fft_channel,
                axes=(1, 2)
            )

            magnitude = np.log1p(
                np.abs(
                    fft_channel
                )
            ).astype(
                np.float32
            )

            scale_blocks.append(
                magnitude.reshape(
                    len(X_view),
                    -1
                )
            )

        fft_by_scale[
            scale
        ] = np.concatenate(
            scale_blocks,
            axis=1
        ).astype(
            np.float32
        )

    return fft_by_scale


def extract_final_matrices_from_nifti(
    nifti_path
):
    """
    Convierte un NIfTI en las tres matrices requeridas:

    - Axial: 6,152 columnas.
    - Axial + sagital: 12,296 columnas.
    - Multiescala: 17,288 columnas.
    """

    centered_volume, metadata = (
        preprocess_volume_xy_variant(
            nifti_path,
            use_restricted_xy=True
        )
    )

    axial_2p5d = np.asarray(
        extract_features_from_volume(
            centered_volume,
            image_size=32
        ),
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    multiview = (
        extract_multiview_projections(
            centered_volume,
            image_size=32
        )
    )

    axial_spatial = np.asarray(
        multiview[
            "axial"
        ],
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    sagittal_spatial = np.asarray(
        multiview[
            "sagittal"
        ],
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    scalar_features = (
        axial_2p5d[
            :,
            3072:
        ]
    )

    axial_multiscale = (
        multiscale_fourier_silent(
            axial_spatial,
            scales=(16, 24, 32)
        )
    )

    sagittal_multiscale = (
        multiscale_fourier_silent(
            sagittal_spatial,
            scales=(16, 24, 32)
        )
    )

    # Axial: espacial + escalares + FFT32
    X_axial_final = np.concatenate(
        [
            axial_2p5d,
            axial_multiscale[32]
        ],
        axis=1
    ).astype(
        np.float32
    )

    # Axial + sagital:
    # espaciales + escalares + FFT32 de ambas vistas
    spatial_multiview_base = (
        np.concatenate(
            [
                axial_spatial,
                sagittal_spatial,
                scalar_features
            ],
            axis=1
        )
    ).astype(
        np.float32
    )

    X_axial_sagittal_final = (
        np.concatenate(
            [
                spatial_multiview_base,
                axial_multiscale[32],
                sagittal_multiscale[32]
            ],
            axis=1
        )
    ).astype(
        np.float32
    )

    # Multiescala en el orden exacto del entrenamiento:
    # base espacial, todas las escalas axiales,
    # todas las escalas sagitales.
    X_multiscale_final = (
        np.concatenate(
            [
                spatial_multiview_base,
                axial_multiscale[16],
                axial_multiscale[24],
                axial_multiscale[32],
                sagittal_multiscale[16],
                sagittal_multiscale[24],
                sagittal_multiscale[32]
            ],
            axis=1
        )
    ).astype(
        np.float32
    )

    expected_shapes = {
        "axial": (1, 6152),
        "axial_sagittal": (
            1,
            12296
        ),
        "multiscale": (
            1,
            17288
        )
    }

    received_shapes = {
        "axial": (
            X_axial_final.shape
        ),
        "axial_sagittal": (
            X_axial_sagittal_final.shape
        ),
        "multiscale": (
            X_multiscale_final.shape
        )
    }

    if (
        received_shapes !=
        expected_shapes
    ):
        raise ValueError(
            f"Formas incorrectas: "
            f"{received_shapes}"
        )

    return {
        "axial": X_axial_final,
        "axial_sagittal": (
            X_axial_sagittal_final
        ),
        "multiscale": (
            X_multiscale_final
        ),
        "metadata": metadata,
        "centered_volume": centered_volume
    }


def predict_final_variant_ensemble(
    X_axial,
    X_axial_sagittal,
    X_multiscale,
    models,
    clip_value=1e-6
):
    """
    Genera las probabilidades finales mediante combinación
    en escala logit de los tres modelos seleccionados.

    Pesos:
    - Axial: 0.14
    - Axial + sagital: 0.21
    - Multiescala: 0.65
    """

    X_axial = np.asarray(
        X_axial,
        dtype=np.float32
    )

    X_axial_sagittal = np.asarray(
        X_axial_sagittal,
        dtype=np.float32
    )

    X_multiscale = np.asarray(
        X_multiscale,
        dtype=np.float32
    )

    n_samples = X_axial.shape[0]

    if (
        X_axial_sagittal.shape[0] !=
        n_samples
        or
        X_multiscale.shape[0] !=
        n_samples
    ):
        raise ValueError(
            "Las tres matrices deben tener "
            "el mismo número de filas."
        )

    expected_columns = {
        "axial": 6152,
        "axial_sagittal": 12296,
        "multiscale": 17288
    }

    received_columns = {
        "axial": X_axial.shape[1],
        "axial_sagittal": (
            X_axial_sagittal.shape[1]
        ),
        "multiscale": (
            X_multiscale.shape[1]
        )
    }

    if (
        received_columns !=
        expected_columns
    ):
        raise ValueError(
            "Columnas incorrectas. "
            f"Esperadas: {expected_columns}. "
            f"Recibidas: {received_columns}."
        )

    axial_probability = (
        models[
            "axial"
        ].predict_proba(
            X_axial
        )[:, 1]
    )

    axial_sagittal_probability = (
        models[
            "axial_sagittal"
        ].predict_proba(
            X_axial_sagittal
        )[:, 1]
    )

    multiscale_probability = (
        models[
            "multiscale"
        ].predict_proba(
            X_multiscale
        )[:, 1]
    )

    axial_probability = np.clip(
        axial_probability,
        clip_value,
        1 - clip_value
    )

    axial_sagittal_probability = np.clip(
        axial_sagittal_probability,
        clip_value,
        1 - clip_value
    )

    multiscale_probability = np.clip(
        multiscale_probability,
        clip_value,
        1 - clip_value
    )

    final_logit = (
        0.14 *
        logit(
            axial_probability
        )
        +
        0.21 *
        logit(
            axial_sagittal_probability
        )
        +
        0.65 *
        logit(
            multiscale_probability
        )
    )

    final_probability = expit(
        final_logit
    )

    final_probability = np.clip(
        final_probability,
        clip_value,
        1 - clip_value
    )

    component_predictions = pd.DataFrame({
        "axial_probability": (
            axial_probability
        ),
        "axial_sagittal_probability": (
            axial_sagittal_probability
        ),
        "multiscale_probability": (
            multiscale_probability
        ),
        "final_probability": (
            final_probability
        )
    })

    return (
        final_probability,
        component_predictions
    )
