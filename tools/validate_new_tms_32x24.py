"""Shared real-capture benchmark; never writes to source TM/dataset files."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GROUPS = ['probe_replay', 'random_amplitude', 'random_phase', 'random_amplitude_phase']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def encode(field, cache):
    """Vectorized holo_SP for logical blocks; preserve relative amplitudes."""
    values, combinations, lut = cache
    field = field / np.max(np.abs(field))
    ids = lut[np.rint(field.real / .01).astype(int) + len(lut)//2,
              np.rint(field.imag / .01).astype(int) + len(lut)//2]
    cells = combinations[ids]
    tiles = np.empty((24, 32, 32, 32), dtype=np.uint8)
    for row in range(8):
        sp = np.roll(cells, -(4*row)%16, axis=-1).reshape(24, 32, 4, 4).swapaxes(-1, -2)
        tiles[:, :, row*4:row*4+4, :] = np.tile(sp, (1, 1, 1, 8))*255
    return tiles.transpose(0, 2, 1, 3).reshape(768, 1024), values[ids].astype(np.complex64)


def prepare(out, count):
    from holograms.generate_LUT import generate_lut
    from holograms.dmd_holograms import holo_SP
    probes = np.load(ROOT/'pregenerated_patterns_8N/probe.npy', mmap_mode='r')
    patterns = np.load(ROOT/'pregenerated_patterns_8N/patterns_pregenerated.npy', mmap_mode='r')
    cache = generate_lut('sp', 4)
    rng = np.random.default_rng(20260910)
    indices = rng.choice(len(probes), count+8, replace=False)
    fields, bitmaps, effective, rows = [], [], [], []
    for k, index in enumerate(indices):
        field = np.array(probes[index])
        _, eff = encode(field, cache)
        fields.append(field); effective.append(eff); bitmaps.append(np.array(patterns[index]))
        rows.append(dict(group='gain_calibration' if k < 8 else 'probe_replay', probe_index=int(index)))
    for group in GROUPS[1:]:
        for k in range(count):
            amp = rng.uniform(0, 1, (24, 32)) if group != 'random_phase' else np.ones((24, 32))
            phase = rng.uniform(-np.pi, np.pi, (24, 32)) if group != 'random_amplitude' else np.zeros((24, 32))
            field = (amp * np.exp(1j*phase)).astype(np.complex64)
            field /= np.max(np.abs(field))
            bitmap, eff = encode(field, cache)
            if k == 0:
                reference = holo_SP(np.kron(field, np.ones((32, 32))), cache[2], cache[1], ds_method='mean')*255
                if not np.array_equal(reference, bitmap):
                    raise RuntimeError('Encoder differs from reference holo_SP')
            fields.append(field); effective.append(eff); bitmaps.append(bitmap)
            rows.append(dict(group=group, probe_index=None))
    # Randomize capture order, with gain samples excluded from all reported test metrics.
    order = rng.permutation(len(rows))
    np.save(out/'inputs.npy', np.asarray(fields)[order])
    np.save(out/'lut_effective_inputs.npy', np.asarray(effective)[order])
    np.save(out/'bitmaps.npy', np.asarray(bitmaps)[order])
    rows = [dict(rows[i], sample_id=k) for k, i in enumerate(order)]
    models = []
    for name in ['reconstructed_field.npy'] + ['reconstructed_field_'+s+'.npy' for s in ['raf21', 'prvbem', 'prvamp', 'wf', 'taf']]:
        path = ROOT/name
        h = np.load(path, mmap_mode='r')
        if h.shape != (16384, 768) or not np.isfinite(h).all():
            raise ValueError('Invalid TM: '+name)
        # Snapshot source matrices so a later GUI reconstruction cannot change the test.
        np.save(out/name, h)
        models.append(dict(name=path.stem.replace('reconstructed_field', 'default_unidentified'),
                           file=name, source=str(path), source_mtime=path.stat().st_mtime,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    dump(out/'manifest.json', dict(seed=20260910, samples=rows, models=models,
         dataset='pregenerated_patterns_8N', dataset_provenance='active dataset; per-TM training provenance not recorded',
         prediction='abs(X @ H.T)**2, ideal requested inputs matching training representation',
         probe_note='Only pure-phase training probes exist; amplitude groups are novel inputs.',
         calibration='8 separate replay probes fit one nonnegative scalar per TM; no per-image fitting'))
    print('PREPARED', out, flush=True)


def capture_all(out):
    import calibrate_v4_32x24 as core
    from hdr_pbr_32x24 import initialize_hardware, set_exposure_us
    from acquisition_snr_32x24 import capture
    if (out/'capture_complete.json').exists():
        raise RuntimeError('Already captured; use analyze to avoid overwriting')
    frames_dir = out/'captures'
    frames_dir.mkdir(exist_ok=False)
    bitmaps = np.load(out/'bitmaps.npy', mmap_mode='r')
    camera = controller = None
    try:
        camera, controller = initialize_hardware(out/'camera')
        exposure = set_exposure_us(camera, core.CAMERA_EXPOSURE_US)
        if not camera.configure_gain(auto_gain='Off', gain=0):
            raise RuntimeError('Cannot lock gain')
        dump(out/'settings.json', dict(exposure_us=exposure, gain_db=float(camera.cam.Gain.GetValue()),
             roi=[camera.roi_x, camera.roi_y, camera.roi_width, camera.roi_height],
             repeats=4, warmup=10, period_us=core.DMD_PICTURE_TIME_US,
             black_reference='all-zero DMD, not shuttered sensor dark'))
        period = core.DMD_PICTURE_TIME_US*1000
        capture(controller, camera, np.zeros((768,1024),np.uint8), 8, frames_dir, 'black_start', period)
        for k, bitmap in enumerate(bitmaps):
            stack = capture(controller, camera, bitmap, 4, frames_dir, 'sample_%03d'%k, period)
            if stack.shape != (4,128,128):
                raise RuntimeError('Unexpected camera ROI')
            print('CAPTURE {}/{} mean={:.3f} max={}'.format(k+1,len(bitmaps),stack.mean(),stack.max()), flush=True)
        capture(controller, camera, np.zeros((768,1024),np.uint8), 8, frames_dir, 'black_end', period)
        dump(out/'capture_complete.json', dict(samples=len(bitmaps), completed=datetime.now().isoformat()))
    finally:
        if controller is not None:
            controller.cleanup()
        if camera is not None:
            camera.cleanup()


def analyze_all(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    if not (out/'capture_complete.json').exists():
        raise RuntimeError('Real capture not complete')
    manifest = json.loads((out/'manifest.json').read_text())
    samples = manifest['samples']
    stacks = np.stack([np.load(out/'captures'/('sample_%03d_frames.npy'%i)) for i in range(len(samples))])
    real = stacks.mean(axis=1).astype(np.float64)
    flat = real.reshape(len(real), -1)
    centered = flat-flat.mean(axis=1,keepdims=True)
    calib = np.array([s['group']=='gain_calibration' for s in samples])
    x = torch.as_tensor(np.load(out/'inputs.npy').reshape(-1,768), device='cuda' if torch.cuda.is_available() else 'cpu')
    metrics, summaries, predictions = [], [], []
    for model in manifest['models']:
        h = torch.as_tensor(np.array(np.load(out/model['file'],mmap_mode='r')),device=x.device)
        pred = ((x@h.T).abs()**2).cpu().numpy().astype(np.float64)
        gain = max(0.,float(np.sum(pred[calib]*flat[calib])/np.sum(pred[calib]**2)))
        pred *= gain
        predictions.append(pred.reshape(real.shape))
        pc = pred-pred.mean(axis=1,keepdims=True)
        corr = np.sum(pc*centered,axis=1)/np.sqrt(np.sum(pc**2,axis=1)*np.sum(centered**2,axis=1))
        nrmse = np.sqrt(np.sum((pred-flat)**2,axis=1)/np.sum(flat**2,axis=1))
        for i,s in enumerate(samples):
            metrics.append(dict(model=model['name'], **s, pearson=float(corr[i]), nrmse=float(nrmse[i]),
                                mean_dn=float(real[i].mean()), saturated_fraction=float((stacks[i]>=255).mean())))
        for group in GROUPS:
            sel=np.array([s['group']==group for s in samples])
            summaries.append(dict(model=model['name'],group=group,n=int(sel.sum()),gain=gain,
                 pearson_mean=float(corr[sel].mean()),pearson_std=float(corr[sel].std(ddof=1)),
                 nrmse_mean=float(nrmse[sel].mean())))
        print(model['name'], 'gain',gain, flush=True)
    for filename,rows in [('metrics.csv',metrics),('summary.csv',summaries)]:
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.save(out/'predictions_scaled.npy',np.asarray(predictions,dtype=np.float32))
    np.save(out/'camera_mean.npy',real.astype(np.float32))
    fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    labels={'default_unidentified':'Latest (unidentified)', 'raf21':'RAF21', 'prvbem':'prVBEM', 'prvamp':'prVAMP', 'wf':'WF', 'taf':'TAF'}
    names=[labels[m['name'].replace('default_unidentified_', '')] for m in manifest['models']]
    for ax,key,title in zip(axes,['pearson_mean','nrmse_mean'],['Spatial Pearson (higher is better)','NRMSE / camera RMS (lower is better)']):
        grid=np.array([[r[key] for r in summaries if r['model']==m['name']] for m in manifest['models']])
        im=ax.imshow(grid,aspect='auto',cmap='viridis' if key=='pearson_mean' else 'viridis_r')
        ax.set_yticks(range(len(names)));ax.set_yticklabels(names)
        ax.set_xticks(range(4));ax.set_xticklabels(['Probe replay','Novel amplitude','Novel phase','Novel amp+phase'],rotation=20,ha='right')
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(4):ax.text(j,i,'%.3f'%grid[i,j],ha='center',va='center',color='white')
        fig.colorbar(im,ax=ax)
    fig.savefig(out/'benchmark.png',dpi=170);plt.close(fig)
    for group in GROUPS:
        # First randomized sample in each group, not selected for favorable performance.
        idx=next(i for i,s in enumerate(samples) if s['group']==group)
        panels=[real[idx]]+[p[idx] for p in predictions]
        fig,axes=plt.subplots(2,len(panels),figsize=(21,6.5),constrained_layout=True)
        vmax=float(np.percentile(real[idx],99.5))
        for j,a in enumerate(panels):
            ax=axes[0,j]
            im=ax.imshow(a,cmap='inferno',vmin=0,vmax=vmax);ax.axis('off')
            title='Real camera (4-frame mean)' if j==0 else names[j-1]
            if j:
                r=next(r for r in metrics if r['model']==manifest['models'][j-1]['name'] and r['sample_id']==idx)
                title+='\nr=%.3f'%r['pearson']
            ax.set_title(title,fontsize=9)
            axes[1,j].imshow(a[48:80,48:80],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
            axes[1,j].axis('off');axes[1,j].set_title('Center 32 x 32 crop',fontsize=8)
        fig.colorbar(im,ax=axes.ravel().tolist(),label='DN; shared scale, clipped at camera P99.5',extend='max')
        fig.suptitle(group+' / sample '+str(idx)+' / upper: full ROI; lower: same fixed crop')
        fig.savefig(out/('comparison_'+group+'.png'),dpi=160);plt.close(fig)
    dump(out/'summary.json',dict(results=summaries,raw_min=float(stacks.min()),raw_max=float(stacks.max()),
         saturated_fraction=float((stacks>=255).mean()),
         caveats=['Probe membership assumes all new TMs use current 8N dataset; no saved per-TM provenance.',
                  'Default TM algorithm unknown. LUT effective fields saved; primary predictions use training-compatible ideal fields.',
                  'NRMSE uses raw camera including background, one gain per model fitted only on separate probe samples.']))
    print('ANALYZED',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','capture','analyze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=24)
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.count<2:parser.error('count must be at least 2')
        args.output.mkdir(parents=True,exist_ok=False)
        prepare(args.output,args.count)
    elif args.stage=='capture':capture_all(args.output)
    else:analyze_all(args.output)
