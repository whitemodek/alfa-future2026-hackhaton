"""Build the flat platform ZIP, excluding credentials, tests and user scratch files."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile

RUNTIME_FILES = ('solution.py', 'agent.py', 'agent_config.py', 'agent_prompts.py',
                 'process_runner.py', 'artifacts.py', 'requirements.txt')


def build(output, source=None):
    source = Path(source or Path(__file__).parent).resolve()
    output = Path(output).resolve()
    names = list(RUNTIME_FILES)
    # The teammate may split their implementation into root ml_*.py modules.
    names.extend(sorted(p.name for p in source.glob('ml_*.py')))
    if output in {source / name for name in names} or output == Path(__file__).resolve():
        raise ValueError('Output must not overwrite solution source')
    for name in names:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing or unsafe runtime file: {name}')
    if sum((source / name).stat().st_size for name in names) > 100_000_000:
        raise ValueError('Uncompressed solution exceeds 100 MB')
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.solution-', suffix='.zip', dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(source / name, arcname=name)
        if temporary.stat().st_size > 100_000_000:
            raise ValueError('ZIP exceeds 100 MB')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(names):
                raise ValueError('ZIP verification failed')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(output), 'files': names, 'bytes': output.stat().st_size,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default='dist/solution.zip')
    print(json.dumps(build(parser.parse_args().output), indent=2))
