from docopt import docopt
from functional_tools import juxt
from pathlib import Path
from typing import Generator
import whisper
from path_utils import walk_paths, is_audio, is_normal_file, make_extension_replacer

label_path = make_extension_replacer("txt")

def audio_labeler(model):
    def label_audio(path: Path) -> str:
        result = model.transcribe(str(path), language="en", task="transcribe", fp16=False)
        return result['text']
    return label_audio

def persist_label(path: Path, text: str):
    with open(label_path(path), 'w') as f:
        f.write(text)
        return path


def log_wrapper(fn):
    def inner(*args, **kwargs):
        result = fn(*args, **kwargs)
        print(f"Function {fn.__name__} called with args: {args}, kwargs: {kwargs}, returned: {result}")
        return result
    return inner

def label_under_path(label_fn, src: Path):
    file_paths = walk_paths(src)
    audio_files = filter(is_audio, file_paths)
    existing_audio_files = filter(is_normal_file, audio_files)
    need_labels = filter(lambda p: not is_normal_file(label_path(p)), existing_audio_files)
    pts = map(juxt(label_path, label_fn), need_labels)
    written_paths = map(lambda pt: persist_label(*pt), pts)
    yield from written_paths

def main(args):
    model = whisper.load_model("large")

    label_fn = audio_labeler(model)

    src = Path(args['<input>'])
    for written in label_under_path(label_fn, src):
        print(written)

USAGE = """
Label Audio Files

Usage:
  label_audio.py <input>

"""
if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)