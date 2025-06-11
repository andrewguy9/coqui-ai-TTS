from docopt import docopt
import glob, os
from pathlib import Path
import shutil
import torch
import json
from train_xtts import find_models, find_runs, model_run_prefix

from TTS.tts.models.xtts import Xtts
from TTS.config import load_config

# find your latest checkpoint (adjust the glob as needed)
def find_best_ckpt(run_dir):
    "Find the config.json and best_model.pth file Paths."
    best = glob.glob(os.path.join(run_dir, "best_model.pth"))
    if not best:
        raise FileNotFoundError(f"No best model found in {run_dir}")
    best_path = run_dir / Path(best[0])
    config = glob.glob(os.path.join(run_dir, "config.json"))
    if not config:
        raise FileNotFoundError(f"No config file found in {run_dir}")
    config_path = run_dir / Path(config[0])
    return config_path, best_path

def export_xtts_weights(run_dir: Path, output_dir: Path):
    config_path, ckpt_path = find_best_ckpt(run_dir)
    # load full checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # remove optimizer state (this typically cuts ~2/3 of the size)
    checkpoint.pop("optimizer", None)
    # optionally remove DVAE/discriminator if present
    for key in list(checkpoint.get("model", {})):
        if "dvae" in key or "disc" in key:
            del checkpoint["model"][key]

    ckpt_path_out = output_dir / "model.pth"
    config_path_out = output_dir / "config.json"

    # save pruned, inference-only checkpoint
    torch.save(checkpoint, ckpt_path_out)
    shutil.copy(config_path, config_path_out)

def export_xtts_weights2(run_dir: Path, out_dir: Path):
    config_path, best_path = find_best_ckpt(run_dir)
    config = load_config(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=best_path)
    config_out_path = out_dir / "config.json"
    best_path_out = out_dir / "model.pth"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_path_out)
    shutil.copy(config_path, config_out_path)

def main(args):
    run_dir = Path(args['<run_dir>'])
    dist_dir = Path(args['<dist_dir>'])

    model_names = args['<model_name>']
    if len(model_names) == 0:
        print(*find_models(run_dir), sep="\n")
        return 0
    for model_name in model_names:
        model_prefix = model_run_prefix(model_name)

        runs = find_runs(model_prefix, run_dir)
        if len(runs) == 0:
            raise FileNotFoundError(f"No runs found for model {model_name} in directory {run_dir}.")
        if len(runs) > 1:
            raise ValueError(f"Multiple runs found for model {model_name} in directory {run_dir}. Please specify a unique run.")
        src_dir = runs[0]

        output_dir = dist_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output directory {output_dir} does not exist or is not a directory.")

        print(f"Exporting XTTS model from {src_dir} to {output_dir}")
        export_xtts_weights2(src_dir, output_dir)

USAGE = """
Export a trained XTTS model from rundir to a specified output directory.

Usage:
    export_model.py <run_dir> <dist_dir> [<model_name>...]
"""

if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)