from docopt import docopt
from pathlib import Path
from typing import Generator
import whisper

def walk_paths(path_str: str) -> Generator[Path, None, None]:
    path = Path(path_str)
    for p in path.rglob('*'):
        yield p

def get_file_extension(path: Path):
    return path.suffix[1:] if path.suffix else ''

def is_audio(path_str):
    ext = get_file_extension(path_str).lower()
    return ext in ('wav', 'mp3')

def is_normal_file(path: Path):
    return path.is_file()

def make_extension_replacer(new_ext: str):
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    def replacer(path: Path) -> Path:
        return path.with_suffix(new_ext)
    return replacer

label_path = make_extension_replacer("txt")

def audio_labeler(model):
    def label_audio(path: Path) -> str:
        result = model.transcribe(str(path))
        return result['text']
    return label_audio

def juxt(*fns):
    """
    Takes a set of functions and returns a fn that is the juxtaposition
    of those fns.  The returned fn takes a variable number of args, and
    returns a vector containing the result of applying each fn to the
    args (left-to-right).
    juxt(a b c)(x) => [a(x), b(x), c(x)]
    """
    def combined(*args, **kwargs):
        return [fn(*args, **kwargs) for fn in fns]
    return combined

def persist_label(path: Path, text: str):
    with open(label_path(path), 'w') as f:
        f.write(text)

def main(args):
    model = whisper.load_model("base")
    label_fn = audio_labeler(model)

    file_paths = walk_paths(args['<input>'])
    audio_files = filter(is_audio, file_paths)
    existing_audio_files = filter(is_normal_file, audio_files)
    need_labels = filter(lambda p: not is_normal_file(label_path(p)), existing_audio_files)
    pts = map(juxt(label_path, label_fn), need_labels)
    list(map(lambda pt: persist_label(*pt), pts))

USAGE = """
Label Audio Files

Usage:
  label_audio.py <input>

"""
if __name__ == "__main__":
    args = docopt(USAGE)
    main(args)