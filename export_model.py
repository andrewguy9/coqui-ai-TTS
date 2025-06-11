from docopt import docopt
import glob, os
from pathlib import Path
import shutil
import torch
import json
from train_xtts import find_runs

from TTS.tts.models.xtts import Xtts

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

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def export_xtts_weights2(run_dir: Path, out_dir: Path):
    config_path, best_path = find_best_ckpt(run_dir)
    config = load_config(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=best_path, map_location="cpu")
    config_out_path = out_dir / "config.json"
    best_path_out = out_dir / "model.pth"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), best_path_out)
    shutil.copy(config_path, config_out_path)

def main(args):
    rundir = Path(args['<rundir>'])
    model_name = args['<model_name>']
    output_dir = Path(args['<output_dir>'])
    output_dir = output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rundir.is_dir():
        raise NotADirectoryError(f"Run directory {rundir} does not exist or is not a directory.")
    print(f"Exporting XTTS model from {rundir} to {output_dir}")
    train_dir = rundir / model_name
    if not train_dir.is_dir():
        raise NotADirectoryError(f"Training directory {train_dir} does not exist or is not a directory.")
    export_xtts_weights2(train_dir, output_dir)

USAGE = """
Export a trained XTTS model from rundir to a specified output directory.

Usage:
    export_model.py <rundir> <model_name> <output_dir>
"""

if __name__ == "__main__":
    args = docopt(USAGE)
