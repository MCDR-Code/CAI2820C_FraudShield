"""Execute a notebook with live output, saving <name>_ran.ipynb beside it."""

# Discover the project from any supported entry point; retain saved-model imports.
import sys as _sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / 'project_paths.py').is_file())
if str(_project) not in _sys.path:
    _sys.path.insert(0, str(_project))
from project_paths import PROJECT_ROOT
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

source = Path(sys.argv[1]).resolve()
target = source.with_name(source.stem + '_ran.ipynb')
if target.exists():
    raise FileExistsError(f'Executed copy already exists: {target}')
notebook = nbformat.read(source, as_version=4)


class LiveClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


client = LiveClient(notebook, timeout=None, kernel_name='python3',
                    resources={'metadata': {'path': str(source.parent)}},
                    on_cell_executed=lambda **k: nbformat.write(notebook, target))
try:
    client.execute()
    print('NOTEBOOK COMPLETED', flush=True)
finally:
    nbformat.write(notebook, target)
