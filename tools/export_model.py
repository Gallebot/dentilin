"""Export a trusted local training checkpoint to a portable inference model."""
from pathlib import Path
import argparse
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description='Exportar modelo de teeth para inferencia')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    # Only the owner's local training checkpoint is loaded with pickle support.
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if int(original.get('num_classes', 0)) != 33:
        raise SystemExit('El checkpoint no pertenece al modelo de teeth esperado.')
    saved = original.get('args', {})
    payload = {'format_version': 1, 'model': original['model'], 'num_classes': 33,
               'args': {'width': int(saved.get('width', 1024)), 'height': int(saved.get('height', 512)),
                        'trainable_backbone_layers': int(saved.get('trainable_backbone_layers', 2))}}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'fdi.pt'
    temporary = target.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(target)
    with target.open('rb') as exported:
        digest = hashlib.file_digest(exported, 'sha256').hexdigest()
    manifest = {'format_version': 1, 'kind': 'teeth', 'file': target.name, 'sha256': digest,
                'bytes': target.stat().st_size, 'num_classes': 33,
                'input_size': [payload['args']['width'], payload['args']['height']], 'url': None}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Exportado {target.name}: {target.stat().st_size / 1e6:.1f} MB; SHA-256 {digest}')

if __name__ == '__main__':
    main()
