import os
import yaml
import argparse
import logging
import subprocess

from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize
from jinja2 import Template

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

sbatch_script_template = """#!/bin/bash

#SBATCH --job-name={{ exp_name }}
#SBATCH --nodes={{ num_nodes }}
#SBATCH --gres=gpu:{{ num_gpus_per_node }}
#SBATCH --ntasks-per-node={{ num_gpus_per_node }}
#SBATCH --cpus-per-task=16
#SBATCH --mem=1560G
#SBATCH --time={{ duration }}
#SBATCH --signal=B:SIGINT@300
#SBATCH --output=outputs/{{ exp_name }}/training.log
#SBATCH --open-mode=append
#SBATCH --partition={{ partition }}
{{ sbatch_overrides }}

unset SLURM_TRES_PER_TASK

export MASTER_ADDR=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n 1)
export MASTER_PORT='12357'

#export PYTHONFAULTHANDLER=1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_VERBOSE_MARK=100
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_CPP_LOG_LEVEL=INFO

git log --pretty=format:'%h' -n 1

srun \\
    python -m vci.train --conf {{ traing_config_file }}
"""


def parse_vars(extra_vars):
    """
    Parses comma seperated key value pair strings into dict.
    """
    vars_list = []
    if extra_vars:
        for i in extra_vars:
            items = i.split("=")
            key = items[0].strip()
            if len(items) > 1:
                value = "=".join(items[1:])
                vars_list.append((key, value))
    return dict(vars_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset list CSV file")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Training configuration file.",
    )
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        help="Experiment name. This will be used to name generated artifacts.",
    )
    parser.add_argument(
        "-n",
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use for this training job.",
    )
    parser.add_argument(
        "-g",
        "--gpus_per_nodes",
        type=int,
        default=4,
        help="Number of GPUs per node",
    )
    parser.add_argument(
        "-r",
        "--reservation",
        dest="reservation",
        type=str,
        default=None,
        help="Slurm reservation to use for this job.",
    )
    parser.add_argument(
        "-p",
        "--partition",
        dest="partition",
        type=str,
        default="gpu_batch,gpu_high_mem,gpu_batch_high_mem,vci_gpu_priority,preemptible",
        help="Slurm partition to use.",
    )
    parser.add_argument(
        "--duration",
        dest="duration",
        type=str,
        default="7-00:00:00",
        help="SLURM job durarion. Pleae refer Slurm documenation for time format",
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="Overwrite config and submit the job.",
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        dest="dryrun",
        action="store_true",
        default=False,
        help="Only generate slurm sbatch script",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        nargs="+",
        default=None,
        help="Values to be overriden for the training.Please refer ./conf/defaults.yaml",
    )

    args = parser.parse_args()

    bind_param = {
        "exp_name": args.exp_name,
        "num_nodes": args.num_nodes,
        "num_gpus_per_node": args.gpus_per_nodes,
        "duration": args.duration,
        "partition": args.partition,
    }

    if args.config:
        bind_param["traing_config_file"] = args.config
    else:
        assert args.exp_name, "Experiment name is required when config is not provided."
        log.info(f"Creating config for {args.exp_name}...")
        trn_conf_dir = Path(f"outputs/{args.exp_name}/conf")
        if not args.force:
            assert not os.path.exists(trn_conf_dir.parent), f"Conf dir {trn_conf_dir.parent.absolute()} already exists."

        overrides = [
            f"experiment.name={args.exp_name}",
            f"experiment.num_nodes={args.num_nodes}",
            f"experiment.num_gpus_per_node={args.gpus_per_nodes}",
        ]

        if args.set:
            log.info(f"Applying overrides: {parse_vars(args.set)}")
            for key, value in parse_vars(args.set).items():
                overrides.append(f"{key}={value}")
            log.info(f"Applying overrides: {overrides}")

        config_dir = Path(os.path.join(os.path.dirname(__file__), "../..", "conf"))
        config_dir = os.path.relpath(config_dir, Path(__file__).parent)
        log.info(config_dir)

        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(
                config_name="defaults.yaml",
                overrides=overrides,
            )
            cfg = OmegaConf.to_container(cfg, resolve=True)

            os.makedirs(trn_conf_dir, exist_ok=True)
            trn_conf_file = Path(f"{trn_conf_dir}/training.yaml")
            with open(trn_conf_file, "w") as file:
                yaml.dump(cfg, file)
            bind_param["traing_config_file"] = trn_conf_file.absolute()

    # SLURM changes
    sbatch_overrides = None
    if args.reservation:
        sbatch_overrides = f"#SBATCH --reservation={args.reservation}\n"

    if sbatch_overrides:
        bind_param["sbatch_overrides"] = sbatch_overrides

    template = Template(sbatch_script_template)
    rendered_script = template.render(bind_param)

    slurm_script = f"outputs/{args.exp_name}/slurm.sh"
    with open(slurm_script, "w") as f:
        f.write(rendered_script)

    if not args.dryrun:
        subprocess.call(["sbatch", slurm_script])
