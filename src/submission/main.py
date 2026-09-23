import os
import tarfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from inference_modelo_2 import (
    extract_final_matrices_from_nifti,
    predict_final_variant_ensemble
)


# En producción será /code_execution.
# La variable permite realizar una prueba local sin modificar
# el comportamiento oficial.
CODE_EXECUTION_ROOT = Path(
    os.environ.get(
        "CODE_EXECUTION_ROOT",
        "/code_execution"
    )
).resolve()

DATA_DIR = (
    CODE_EXECUTION_ROOT /
    "data"
)

NIFTI_DIR = (
    DATA_DIR /
    "niftis"
)

SUBMISSION_FORMAT_PATH = (
    DATA_DIR /
    "submission_format.csv"
)

OUTPUT_PATH = (
    CODE_EXECUTION_ROOT /
    "submission.csv"
)


from inference_roi_asymmetry import (
    predict_roi_probability
)

from inference_efficientnet_multiview import (
    load_efficientnet_multiview_model,
    predict_efficientnet_probability
)

from inference_cnn3d import (
    load_cnn3d_models,
    predict_cnn3d_hybrid_probability
)

SCRIPT_DIR = Path(
    __file__
).resolve().parent

MODEL_PATH = (
    SCRIPT_DIR /
    "final_variant_models.joblib"
)

ROI_MODEL_PATH = (
    SCRIPT_DIR /
    "roi_asymmetry_linear_parameters.npz"
)

EFFICIENTNET_ENCODER_PATH = (
    SCRIPT_DIR /
    "efficientnet_b0_encoder_state_dict.pt"
)

EFFICIENTNET_LINEAR_PARAMETERS_PATH = (
    SCRIPT_DIR /
    "efficientnet_multiview_linear_parameters.npz"
)

CNN3D_STRATIFIED_PATHS = [
    SCRIPT_DIR / f"cnn3d_strat_fold_{fold}.pt"
    for fold in range(1, 6)
]

CNN3D_GROUPED_PATHS = [
    SCRIPT_DIR / f"cnn3d_group_fold_{fold}.pt"
    for fold in range(1, 6)
]


def uid_from_path(path):
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    if name.endswith(".nii"):
        return name[:-4]

    raise ValueError(
        "Extensión NIfTI no reconocida."
    )


def locate_nifti_files(expected_uids):
    """
    Localiza cada NIfTI con las mismas rutas utilizadas
    por la entrega sub_v5 que funcionó.
    """

    uid_to_path = {}

    for uid in expected_uids:

        uid = str(uid)

        candidates = (
            DATA_DIR /
            "nifts" /
            f"{uid}.nii.gz",

            DATA_DIR /
            "niftis" /
            f"{uid}.nii.gz",

            DATA_DIR /
            f"{uid}.nii.gz",

            DATA_DIR /
            "nifts" /
            f"{uid}.nii",

            DATA_DIR /
            "niftis" /
            f"{uid}.nii",

            DATA_DIR /
            f"{uid}.nii",
        )

        image_path = next(
            (
                path
                for path in candidates
                if path.is_file()
            ),
            None
        )

        if image_path is None:

            raise FileNotFoundError(
                "A required NIfTI examination "
                "could not be located."
            )

        uid_to_path[
            uid
        ] = image_path

    return uid_to_path

def main():

    print(
        "Starting inference — roi-efficientnet-cnn3d-optimized-v2.",
        flush=True
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el archivo del modelo."
        )

    if not SUBMISSION_FORMAT_PATH.exists():
        raise FileNotFoundError(
            "No se encontró submission_format.csv."
        )

    if not NIFTI_DIR.exists():
        raise FileNotFoundError(
            "No se encontró el directorio niftis."
        )

    models = joblib.load(
        MODEL_PATH
    )

    submission_format = pd.read_csv(
        SUBMISSION_FORMAT_PATH
    )

    expected_columns = [
        "uid",
        "is_pathologic"
    ]

    if (
        submission_format.columns.tolist()
        != expected_columns
    ):
        raise ValueError(
            "Las columnas de submission_format.csv "
            "no coinciden con el formato esperado."
        )

    if submission_format[
        "uid"
    ].duplicated().any():

        raise ValueError(
            "submission_format.csv contiene UID duplicados."
        )

    ordered_uids = (
        submission_format[
            "uid"
        ]
        .astype(str)
        .tolist()
    )

    if not ROI_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo ROI."
        )

    with np.load(
        ROI_MODEL_PATH,
        allow_pickle=False
    ) as saved_roi:

        roi_parameters = {
            name: saved_roi[name].copy()
            for name in saved_roi.files
        }

    if not EFFICIENTNET_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el encoder EfficientNet."
        )

    if not EFFICIENTNET_LINEAR_PARAMETERS_PATH.exists():
        raise FileNotFoundError(
            "No se encontraron los parámetros "
            "lineales EfficientNet."
        )

    (
        efficientnet_encoder,
        efficientnet_parameters
    ) = load_efficientnet_multiview_model(
        encoder_path=EFFICIENTNET_ENCODER_PATH,
        linear_parameters_path=(
            EFFICIENTNET_LINEAR_PARAMETERS_PATH
        )
    )

    cnn3d_models = load_cnn3d_models(
        stratified_paths=CNN3D_STRATIFIED_PATHS,
        grouped_paths=CNN3D_GROUPED_PATHS
    )

    uid_to_path = (
        locate_nifti_files(
            ordered_uids
        )
    )

    missing_uids = [
        uid
        for uid in ordered_uids
        if uid not in uid_to_path
    ]

    if missing_uids:
        raise FileNotFoundError(
            "Faltan archivos NIfTI requeridos."
        )

    probabilities = np.empty(
        len(ordered_uids),
        dtype=np.float64
    )

    for index, uid in enumerate(
        ordered_uids
    ):

        extracted = (
            extract_final_matrices_from_nifti(
                uid_to_path[
                    uid
                ]
            )
        )

        probability, _ = (
            predict_final_variant_ensemble(
                X_axial=(
                    extracted[
                        "axial"
                    ]
                ),
                X_axial_sagittal=(
                    extracted[
                        "axial_sagittal"
                    ]
                ),
                X_multiscale=(
                    extracted[
                        "multiscale"
                    ]
                ),
                models=models
            )
        )

        previous_probability = float(
            probability[0]
        )

        roi_probability, _ = (
            predict_roi_probability(
                volume=extracted[
                    "centered_volume"
                ],
                parameters=roi_parameters
            )
        )

        (
            cnn_probability,
            _,
            _
        ) = predict_efficientnet_probability(
            centered_volume=extracted[
                "centered_volume"
            ],
            encoder=efficientnet_encoder,
            linear_parameters=(
                efficientnet_parameters
            )
        )

        previous_probability = float(
            np.clip(
                previous_probability,
                1e-6,
                1 - 1e-6
            )
        )

        roi_probability = float(
            np.clip(
                roi_probability,
                1e-6,
                1 - 1e-6
            )
        )

        cnn_probability = float(
            np.clip(
                cnn_probability,
                1e-6,
                1 - 1e-6
            )
        )

        previous_logit = np.log(
            previous_probability /
            (
                1.0 -
                previous_probability
            )
        )

        roi_logit = np.log(
            roi_probability /
            (
                1.0 -
                roi_probability
            )
        )

        cnn_logit = np.log(
            cnn_probability /
            (
                1.0 -
                cnn_probability
            )
        )

        current_ensemble_logit = (
            0.50 * previous_logit +
            0.50 * roi_logit
        )

        previous_full_ensemble_logit = (
            0.625 * current_ensemble_logit +
            0.375 * cnn_logit
        )

        (
            _,
            cnn3d_diagnostics
        ) = predict_cnn3d_hybrid_probability(
            nifti_path=uid_to_path[uid],
            model_bundle=cnn3d_models
        )

        # La optimización OOF corresponde a la familia
        # CNN3D estratificada, no a la mezcla híbrida.
        cnn3d_stratified_probability = float(
            np.clip(
                cnn3d_diagnostics[
                    "stratified_probability"
                ],
                1e-6,
                1 - 1e-6
            )
        )

        cnn3d_stratified_logit = np.log(
            cnn3d_stratified_probability /
            (
                1.0 -
                cnn3d_stratified_probability
            )
        )

        # Mezcla optimizada en espacio logit.
        uncalibrated_final_logit = (
            0.634346000000
            * previous_full_ensemble_logit
            +
            0.365654000000
            * cnn3d_stratified_logit
        )

        # Calibración afín desplegada. Los parámetros se
        # reajustaron sobre el OOF completo después de la
        # selección cross-fitted; 0.362112 es la estimación
        # cross-fitted y 0.361771 el ajuste optimizado full-OOF.
        final_logit = (
            uncalibrated_final_logit
            + 0.131114806797781
        ) / 0.761293538704342

        probabilities[
            index
        ] = float(
            1.0 /
            (
                1.0 +
                np.exp(
                    -final_logit
                )
            )
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Se generaron predicciones no finitas."
        )

    if not (
        (probabilities >= 0.0).all()
        and
        (probabilities <= 1.0).all()
    ):

        raise ValueError(
            "Se generaron probabilidades fuera de rango."
        )

    submission = (
        submission_format.copy()
    )

    submission[
        "uid"
    ] = submission[
        "uid"
    ].astype(str)

    submission[
        "is_pathologic"
    ] = probabilities

    if (
        submission.columns.tolist()
        != expected_columns
    ):
        raise RuntimeError(
            "La salida no tiene las columnas requeridas."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    submission.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "Inference completed successfully.",
        flush=True
    )


if __name__ == "__main__":
    main()
