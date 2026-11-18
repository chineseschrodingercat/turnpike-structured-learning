from pathlib import Path
import argparse,zipfile,json,hashlib
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('directories',nargs='+',type=Path);ap.add_argument('--out',type=Path,default=Path('MPC_server_results.zip'));args=ap.parse_args()
files={}
folder_names=[p.resolve().name for p in args.directories]
if len(folder_names)!=len(set(folder_names)):ap.error('Result directories must have distinct final folder names to avoid archive collisions.')
for folder in args.directories:
    folder=folder.resolve()
    if not folder.is_dir():ap.error('Missing results directory: '+str(folder))
    for p in folder.rglob('*'):
        if p.is_file():files['results/'+folder.name+'/'+p.relative_to(folder).as_posix()]=p
for name in ['RUN_PROTOCOL.json','MANIFEST_SHA256.json','SOURCE_PROVENANCE.json']:
    files['package_identity/'+name]=ROOT/name
target=args.out.resolve()
if target.exists():ap.error('Output exists; choose a new --out filename.')
manifest=[{'path':k,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in sorted(files.items())]
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(files.items()):z.write(p,name)
    z.writestr('RETURN_MANIFEST.json',json.dumps(manifest,indent=2))
print(target)
