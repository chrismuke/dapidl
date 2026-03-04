"""AWS Auto-Scaler for ClearML.

Forked from clearml/examples/services/aws-autoscaler/aws_autoscaler.py
(commit eef3ae28) to add iam_arn/iam_name support. ClearML's task.connect()
only syncs keys present in the local dict template, so the upstream script
silently ignores iam_arn/iam_name stored in task hyperparameters.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

from clearml import Task
from clearml.automation.auto_scaler import AutoScaler, ScalerConfig
from clearml.automation.aws_driver import AWSDriver
from clearml.config import running_remotely


default_config = {
    "hyper_params": {
        "git_user": "",
        "git_pass": "",
        "cloud_credentials_key": "",
        "cloud_credentials_secret": "",
        "cloud_credentials_region": None,
        "cloud_credentials_token": "",
        "use_credentials_chain": False,
        "default_docker_image": "nvidia/cuda",
        "max_idle_time_min": 15,
        "polling_interval_time_min": 5,
        "max_spin_up_time_min": 30,
        "workers_prefix": "dynamic_worker",
        "cloud_provider": "",
        # IAM instance profile — these were missing from upstream, causing
        # AWSDriver.from_config() to never receive them via task.connect()
        "iam_arn": "",
        "iam_name": "",
        "use_iam_instance_profile": False,
    },
    "configurations": {
        "resource_configurations": None,
        "queues": None,
        "extra_trains_conf": "",
        "extra_clearml_conf": "",
        "extra_vm_bash_script": "",
        "docker_force_pull": False,
    },
}


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--run",
        help="Run the autoscaler after wizard finished",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--remote",
        help="Run the autoscaler as a service, launch on the `services` queue",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--config-file",
        help="Configuration file name",
        type=Path,
        default=Path("aws_autoscaler.yaml"),
    )
    args = parser.parse_args()

    if running_remotely():
        conf = default_config
    else:
        if args.config_file.exists():
            import yaml

            with args.config_file.open("r") as f:
                conf = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            print(
                "Error: config file '{}' not found. "
                "Create it or use the upstream ClearML wizard.".format(args.config_file)
            )
            return

    task = Task.init(
        project_name="DevOps",
        task_name="AWS Auto-Scaler",
        task_type=Task.TaskTypes.service,
    )
    task.connect(conf["hyper_params"])
    configurations = conf["configurations"]
    configurations.update(
        json.loads(task.get_configuration_object(name="General") or "{}")
    )
    task.set_configuration_object(
        name="General", config_text=json.dumps(configurations, indent=2)
    )

    if args.remote or args.run:
        print(
            "Running AWS auto-scaler as a service\nExecution log {}".format(
                task.get_output_log_web_page()
            )
        )

    if args.remote:
        task.execute_remotely(queue_name="services")

    driver = AWSDriver.from_config(conf)
    conf = ScalerConfig.from_config(conf)
    autoscaler = AutoScaler(conf, driver)
    if running_remotely() or args.run:
        autoscaler.start()


if __name__ == "__main__":
    main()
