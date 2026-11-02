"""Execute a notebook with live output, saving <name>_ran.ipynb beside it."""
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
