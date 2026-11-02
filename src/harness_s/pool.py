"""Relocatable, content-checked sibling harness archive."""
from pathlib import Path
from .io import read_json, digest


class HarnessPool:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.catalog = read_json(self.root / 'harness_pool/appworld/catalog.json')
        self.members = {item['id']: item for item in self.catalog['members']}
        self.h0 = self.catalog['h0']

    def verify(self):
        for member in self.members.values():
            for path, expected in member['sha256'].items():
                if digest(self.root / path) != expected:
                    raise ValueError(f'Changed harness artifact: {path}')
        return len(self.members)

    def render(self, destination):
        destination = Path(destination).resolve()
        bindings = {}
        for hid, member in self.members.items():
            source = self.root / 'harness_pool/appworld' / member['path']
            target = destination / hid / 'harness.yml'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text().replace('{{HARNESS_S_ROOT}}', str(self.root)))
            bindings[hid] = dict(id=hid, path=str(target), sha256=digest(target))
        return bindings
