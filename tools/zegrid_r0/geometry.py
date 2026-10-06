#!/usr/bin/env python3
"""ZM-ZEGRID-R0 diagnostic, NOT production. Header-only, undistorted TAN only.

python tools/zegrid_r0/geometry.py --input /path/lights --output result.json
No pixel access/stacking. Existing project dependency Shapely is reused.
Coordinates: zero-origin pixel centres; integer half-open slices have edges x-.5.
Source crops are geometric estimates, not a validated FITS extraction engine.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject.mosaicking import find_optimal_celestial_wcs
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

@dataclass(frozen=True)
class Frame:
    key: str
    wcs: WCS  # private snapshot; dataclass freeze does not freeze Astropy internals
    shape: tuple[int, int]
    header_sha256: str = ''
    instrument: str = ''

@dataclass(frozen=True)
class Canvas:
    wcs: WCS
    width: int
    height: int
    resolution_deg: float


def check_wcs(w):
    if w.pixel_n_dim != 2 or not w.has_celestial or w.has_distortion:
        raise ValueError('witness supports undistorted 2D celestial TAN only')
    if tuple(w.wcs.ctype) != ('RA---TAN', 'DEC--TAN') or w.wcs.get_pv():
        raise ValueError('witness requires RA/DEC TAN, no PV terms')


def read_frames(root):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.fit','.fits','.fts'))
    if not paths:
        raise ValueError('no FITS files')
    frames, rejected, seen = [], [], set()
    for p in paths:
        try:
            resolved = p.resolve()
            if resolved in seen:
                raise ValueError('duplicate resolved path')
            seen.add(resolved)
            # getheader never accesses HDU.data; no full source pixel read.
            h = fits.getheader(p, 0)
            if h.get('NAXIS') != 2:
                raise ValueError('witness accepts 2D primary images only; CHW/HWC must be explicit in R1')
            w = WCS(h)
            check_wcs(w)
            shape = (int(h['NAXIS2']), int(h['NAXIS1']))
            if min(shape) <= 0:
                raise ValueError('nonpositive shape')
            frames.append(Frame(p.relative_to(root).as_posix(), w, shape,
                hashlib.sha256(h.tostring().encode()).hexdigest(), str(h.get('INSTRUME',''))))
        except Exception as exc:
            rejected.append({'path':p.relative_to(root).as_posix(), 'reason':str(exc)})
    if not frames:
        raise ValueError('no supported WCS; exclusions: '+str(rejected[:5]))
    return frames, rejected


def boundary(shape):
    h,w = shape
    return np.array([[-.5,-.5],[w-.5,-.5],[w-.5,h-.5],[-.5,h-.5]])


def project_points(points, source, target):
    sky = source.pixel_to_world(points[:,0],points[:,1])
    x,y = target.world_to_pixel(sky)
    xy = np.column_stack((x,y))
    if not np.isfinite(xy).all():
        raise ValueError('nonfinite projection; no geometry fallback')
    # Reject opposite TAN hemisphere, where apparent finite inverse is not valid.
    back = target.pixel_to_world(x,y)
    if np.any(back.separation(sky).deg > 1e-7):
        raise ValueError('projection roundtrip/hemisphere failure')
    return xy


def polygon(frame, target):
    p = Polygon(project_points(boundary(frame.shape), frame.wcs, target))
    if not p.is_valid or p.area <= 0:
        raise ValueError('invalid projected footprint')
    return p


def make_canvas(frames):
    frames = sorted(frames, key=lambda f:f.key)
    if len({f.key for f in frames}) != len(frames):
        raise ValueError('duplicate frame identity')
    for f in frames:
        check_wcs(f.wcs)
    # Correct numeric API, unlike legacy _extract_pixel_scale_deg's Quantity assumption.
    resolution = float(np.median([np.mean(np.abs(proj_plane_pixel_scales(f.wcs))) for f in frames]))
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('invalid scale')
    w,_ = find_optimal_celestial_wcs([(f.shape,f.wcs) for f in frames],
            resolution=resolution*u.deg, projection='TAN',auto_rotate=True)
    bounds=np.array([polygon(f,w).bounds for f in frames])
    # Shift is baked once into WCS. Canvas pixels have no hidden offset.
    x0,y0=np.floor(bounds[:,:2].min(axis=0)+.5).astype(int)
    x1,y1=np.ceil(bounds[:,2:].max(axis=0)+.5).astype(int)
    w=copy.deepcopy(w)
    w.wcs.crpix-=np.array([x0,y0])
    w.array_shape=(int(y1-y0),int(x1-x0))
    return Canvas(w,int(x1-x0),int(y1-y0),resolution)


def cells(canvas,nx,ny,halo):
    if not 1 <= nx <= canvas.width or not 1 <= ny <= canvas.height or halo < 0:
        raise ValueError('invalid layout or halo')
    # Integer partition, no rounding gaps; edge cells may differ by one pixel.
    for iy in range(ny):
        for ix in range(nx):
            x0,x1=ix*canvas.width//nx,(ix+1)*canvas.width//nx
            y0,y1=iy*canvas.height//ny,(iy+1)*canvas.height//ny
            core=(x0,y0,x1,y1)
            patch=(max(0,x0-halo),max(0,y0-halo),min(canvas.width,x1+halo),min(canvas.height,y1+halo))
            yield f'r{iy:04d}c{ix:04d}',core,patch


def rect(bounds):
    x0,y0,x1,y1=bounds
    return box(x0-.5,y0-.5,x1-.5,y1-.5)


def source_crop(frame,canvas,intersection,margin):
    if intersection.is_empty or intersection.area <= 0:
        return None
    if intersection.geom_type != 'Polygon':
        raise ValueError('nonconvex/disconnected intersection not qualified')
    xy=project_points(np.array(intersection.exterior.coords),canvas.wcs,frame.wcs)
    h,w=frame.shape
    # Conservative rectangular bound plus explicit source-pixel interpolation hypothesis.
    lo=np.floor(xy.min(axis=0)-margin).astype(int)
    hi=np.ceil(xy.max(axis=0)+margin).astype(int)+1
    return (max(0,int(lo[0])),max(0,int(lo[1])),min(w,int(hi[0])),min(h,int(hi[1])))


def area(b):
    return max(0,b[2]-b[0])*max(0,b[3]-b[1])


def stats(values):
    a=np.asarray(values)
    return dict(min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()))


def simulate(frames,canvas,nx,ny,halo,source_margin=2):
    frames=sorted(frames,key=lambda f:f.key)
    polys=[polygon(f,canvas.wcs) for f in frames]
    union=unary_union(polys)
    records=[]
    source_rects={f.key:[] for f in frames}
    full=local=patch_area=core_area=0
    coverage=[]
    bbox_false=0
    for cid,core,patch in cells(canvas,nx,ny,halo):
        cr,pr=rect(core),rect(patch)
        members=[];pmembers=[]
        for f,p in zip(frames,polys):
            if p.intersection(cr).area > 1e-8:
                members.append(f.key)
            inter=p.intersection(pr)
            if inter.area > 1e-8:
                pmembers.append(f.key)
                crop=source_crop(f,canvas,inter,source_margin)
                local+=area(crop)
                source_rects[f.key].append(box(*crop))
                full+=f.shape[0]*f.shape[1]
            elif box(*p.bounds).intersection(pr).area > 1e-8:
                bbox_false+=1
        ca=area(core)
        coverage.append(float(union.intersection(cr).area/ca))
        core_area+=ca;patch_area+=area(patch)
        records.append(dict(cell_id=cid,core=core,patch=patch,frames=members,patch_frames=pmembers))
    unique=sum(unary_union(v).area for v in source_rects.values() if v)
    count=[len(r['frames']) for r in records]
    pc=[len(r['patch_frames']) for r in records]
    # Sampled geometric multiplicity, not positive canonical support or valid-pixel coverage.
    xx,yy=np.meshgrid((np.arange(80)+.5)*canvas.width/80-.5,(np.arange(80)+.5)*canvas.height/80-.5)
    from shapely import contains_xy
    density=sum(contains_xy(p,xx,yy).astype(int) for p in polys)
    vals,cts=np.unique(density,return_counts=True)
    return dict(nx=nx,ny=ny,halo_target_px=halo,source_margin_px=source_margin,
        cell_width=stats([r['core'][2]-r['core'][0] for r in records]),
        cell_height=stats([r['core'][3]-r['core'][1] for r in records]),
        total_cells=len(records),nonempty_cells=sum(n>0 for n in count),empty_cells=sum(n==0 for n in count),
        frames_per_cell=stats(count),frames_per_patch=stats(pc),
        cell_area=core_area,patch_area=patch_area,halo_overhead=patch_area/core_area-1,
        estimated_source_pixels=local,unique_source_crop_pixels=float(unique),
        source_pixel_redundancy=local/unique if unique else None,
        source_pixels_once=sum(f.shape[0]*f.shape[1] for f in frames),
        full_source_per_patch_pixels=full,full_source_vs_local_ratio=full/local if local else None,
        bbox_only_false_positive_memberships=bbox_false,
        geometric_covered_fraction=union.intersection(rect((0,0,canvas.width,canvas.height))).area/core_area,
        cell_covered_area_fraction=stats(coverage),
        geometric_density_80x80={str(v):int(c) for v,c in zip(vals,cts)},cells=records)


def run(frames,layouts=None,halos=(0,8,32)):
    canvas=make_canvas(frames)
    widths=[];heights=[]
    for f in frames:
        x0,y0,x1,y1=polygon(f,canvas.wcs).bounds
        widths.append(x1-x0);heights.append(y1-y0)
    # Shape-aware comparison, NOT a frozen Auto product policy.
    ratio=(canvas.width/np.median(widths))/(canvas.height/np.median(heights))
    if layouts is None:
        layouts=sorted(set((max(1,round(math.sqrt(n*ratio))),max(1,round(math.sqrt(n/ratio)))) for n in (4,16,36,64)))
    return dict(schema='ZM-ZEGRID-R0-geometry-v1',scope='header-only undistorted TAN geometry; no science/time claim',
        frame_count=len(frames),canvas=dict(width=canvas.width,height=canvas.height,resolution_deg=canvas.resolution_deg,
            wcs_header=canvas.wcs.to_header().tostring()),
        median_source_width=float(np.median(widths)),median_source_height=float(np.median(heights)),
        frames=[dict(key=f.key,shape=f.shape,header_sha256=f.header_sha256,instrument=f.instrument) for f in sorted(frames,key=lambda f:f.key)],
        layouts=[simulate(frames,canvas,nx,ny,h) for nx,ny in layouts for h in halos])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--halos',default='0,8,32')
    args=parser.parse_args()
    frames,rejected=read_frames(args.input)
    result=run(frames,halos=tuple(int(v) for v in args.halos.split(',')))
    result['input_root']=str(args.input.resolve())
    result['rejected']=rejected
    result['real_corpus_status']='PASS' if not rejected else 'PARTIAL'
    import astropy,reproject,shapely
    result['versions']=dict(numpy=np.__version__,astropy=astropy.__version__,reproject=reproject.__version__,shapely=shapely.__version__)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('frames','layouts','canvas')},indent=2))
    print('canvas',result['canvas']['width'],result['canvas']['height'])
    for row in result['layouts']:
        print(row['nx'],row['ny'],row['halo_target_px'],row['nonempty_cells'],row['frames_per_patch'],row['source_pixel_redundancy'],row['full_source_vs_local_ratio'])

if __name__=='__main__':
    main()
