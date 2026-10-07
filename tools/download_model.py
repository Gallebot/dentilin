"""Download or verify the exact model described by models/manifest.json."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def verify(path, manifest):
    if not path.is_file():
        raise ValueError(f'No existe el modelo: {path}')
    if path.stat().st_size != manifest['bytes']:
        raise ValueError('El tamaño del modelo no coincide con el manifiesto.')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('El SHA-256 del modelo no coincide con el manifiesto.')

def main():
    parser = argparse.ArgumentParser(description='Descargar o verificar el modelo')
    parser.add_argument('--url', default=os.environ.get('DENTILIN_MODEL_URL'))
    parser.add_argument('--models-dir', type=Path, default=Path(os.environ.get('DENTILIN_MODEL_DIR', ROOT / 'models')))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    # The expected checksum comes from the app release, never from the download URL.
    manifest = json.loads((ROOT / 'models/manifest.json').read_text(encoding='utf-8'))
    filename = manifest['file']
    if Path(filename).name != filename:
        raise SystemExit('Nombre de modelo inválido en el manifiesto.')
    target = args.models_dir / filename
    try:
        if args.verify_only:
            verify(target, manifest)
            print('Modelo verificado.')
            return
        url = args.url or manifest.get('url')
        if not url or urlparse(url).scheme != 'https':
            raise ValueError('Indica --url con la URL HTTPS del modelo publicado en la release.')
        args.models_dir.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.pt.download')
        try:
            with urlopen(url, timeout=120) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            verify(temporary, manifest)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Modelo descargado y verificado: {target}')
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))

if __name__ == '__main__':
    main()
