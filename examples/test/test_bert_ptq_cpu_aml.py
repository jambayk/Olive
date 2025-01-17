# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from utils import check_output, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "bert"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


@pytest.mark.parametrize(
    "olive_test_knob",
    [
        # aml system test
        ("bert_ptq_cpu.json", "tpe", "joint", "aml_system"),
        # aml model test in local system
        ("bert_ptq_cpu_aml.json", False, None, "local_system"),
        # aml model test in aml system
        ("bert_ptq_cpu_aml.json", False, None, "aml_system"),
    ],
)
def test_bert(olive_test_knob):
    # olive_config: (config_json_path, search_algorithm, execution_order, system)
    # bert_ptq_cpu.json: use huggingface model id
    # bert_ptq_cpu_aml.json: use aml model path
    from olive.workflows import run as olive_run

    olive_config = patch_config(*olive_test_knob)
    if olive_test_knob[3] == "aml_system":
        # remove the invalid OpenVINOExecutionProvider for bert aml system.
        olive_config["engine"]["execution_providers"] = ["CPUExecutionProvider"]
        # remove goal for aml system since sometimes the aml job will be reused.
        # If the jobs perf cannot meet the goal, the test will fail definitely.
        metrics = olive_config["evaluators"]["common_evaluator"]["metrics"]
        metrics[0]["sub_types"][0].pop("goal", None)
        metrics[1]["sub_types"][0].pop("goal", None)

    output = olive_run(olive_config)
    check_output(output)
