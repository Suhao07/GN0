from typing import List, Optional, Union

from GN_Bench.config.default import CONFIG_FILE_SEPARATOR
from GN_Bench.config.default import Config as CN
from GN_Bench.config.default import get_config as get_base_config

_C = get_base_config()
_C.defrost()

# GN0 extends GN-Bench only with the metrics used by the InteriorGS evaluator.
# Legacy VLN-CE panorama, RxR, GPS, and point-navigation defaults intentionally
# stay out of this package.
_C.TASK.PATH_LENGTH = CN()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
_C.TASK.ORACLE_SUCCESS = CN()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 3.0
_C.TASK.STEPS_TAKEN = CN()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"


# ----------------------------------------------------------------------------
# GN0 EVALUATION CONFIG
# ----------------------------------------------------------------------------
_EVAL_C = CN()
_EVAL_C.BASE_TASK_CONFIG_PATH = ""
_EVAL_C.TASK_CONFIG = CN()
_EVAL_C.CMD_TRAILING_OPTS = []
_EVAL_C.EVAL = CN()
_EVAL_C.EVAL.IDENTIFICATION = "gs"
_EVAL_C.EVAL.STALL_PATIENCE = 16
_EVAL_C.EVAL.EARLY_STOP_STEPS = 500
_EVAL_C.EVAL.RENDERER = CN()
_EVAL_C.EVAL.RENDERER.VIDEO_FPS = 10.0
_EVAL_C.EVAL.RENDERER.VIDEO_CODEC = "libx264"
_EVAL_C.EVAL.RENDERER.SMOOTH_WINDOW = 9
_EVAL_C.EVAL.RENDERER.RESAMPLE_STEP_M = 0.05
_EVAL_C.EVAL.RENDERER.HEADING_LOOKAHEAD_M = 0.5
_EVAL_C.EVAL.RENDERER.YAW_SMOOTH_WINDOW = 9


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    inline_config: Optional[CN] = None,
) -> CN:
    """Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if inline_config is not None and len(inline_config) > 0:
        config.merge_from_other_cfg(inline_config)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Create the GN0 evaluation config without VLN-CE baseline defaults."""
    config = _EVAL_C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.TASK_CONFIG = get_extended_config(
        config.BASE_TASK_CONFIG_PATH or None,
        inline_config=config.TASK_CONFIG,
    )
    config.freeze()
    return config
