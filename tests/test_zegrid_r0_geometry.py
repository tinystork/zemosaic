"""R0 geometry-only witness and executable historical claims; no production changes."""
from pathlib import Path
import copy
import importlib.util
import sys
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from shapely.geometry import Polygon, box
from zemosaic import grid_mode as gm
from zemosaic.core.canonical_engine import CanonicalStackRequest, run_canonical_stack
from zemosaic.core.canonical_support import make_footprint_taper

spec=importlib.util.spec_from_file_location('zegrid_r0_geometry',Path(__file__).parents[1]/'tools/zegrid_r0/geometry.py')
g=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=g
spec.loader.exec_module(g)


def frame(key='a',ra=10.,dec=30.,angle=0.,shape=(40,60)):
    w=WCS(naxis=2)
    w.wcs.ctype=['RA---TAN','DEC--TAN']
    w.wcs.crval=[ra,dec]
    w.wcs.crpix=[shape[1]/2,shape[0]/2]
    a=np.deg2rad(angle)
    w.wcs.cd=np.array([[-np.cos(a),np.sin(a)],[np.sin(a),np.cos(a)]])*.001
    w.array_shape=shape
    return g.Frame(key,w,shape)

@pytest.mark.parametrize('coords',[(10.,10.04),(359.98,.02)])
def test_canvas_layout_membership_permutation(coords):
    fs=[frame('z',coords[0],angle=27),frame('a',coords[1],angle=-15),frame('b',coords[0],dec=30.02)]
    expected=g.run(fs,layouts=[(3,2)],halos=(0,8))
    rng=np.random.default_rng(42)
    for order in (list(reversed(fs)),[fs[i] for i in rng.permutation(3)]):
        assert g.run(order,layouts=[(3,2)],halos=(0,8))==expected

@pytest.mark.parametrize('shape,layout', [((31,99),(7,3)),((19,701),(21,2)),((301,17),(2,19))])
def test_partition_and_halo_edges(shape,layout):
    f=frame(shape=shape)
    c=g.make_canvas([f]);seen=np.zeros((c.height,c.width),int)
    for cid,core,patch in g.cells(c,*layout,halo=8):
        x0,y0,x1,y1=core;px0,py0,px1,py1=patch
        seen[y0:y1,x0:x1]+=1
        assert 0<=px0<=x0<x1<=px1<=c.width
        assert 0<=py0<=y0<y1<=py1<=c.height
    assert np.all(seen==1)


def test_bbox_false_positive():
    diamond=Polygon([(0,5),(5,0),(10,5),(5,10)])
    patch=box(0,0,1,1)
    assert box(*diamond.bounds).intersection(patch).area==1
    assert diamond.intersection(patch).area==0


def test_source_crops_cover_dense_target_samples():
    fs=[frame('a',angle=32),frame('b',ra=10.02,angle=-11)]
    c=g.make_canvas(fs)
    for f in fs:
        p=g.polygon(f,c.wcs)
        for _,_,patch in g.cells(c,3,2,5):
            inter=p.intersection(g.rect(patch))
            crop=g.source_crop(f,c,inter,2)
            if crop is None: continue
            x0,y0,x1,y1=patch
            xx,yy=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1))
            from shapely import contains_xy
            inside=contains_xy(p,xx,yy)
            xy=g.project_points(np.column_stack((xx[inside],yy[inside])),c.wcs,f.wcs)
            assert np.all(xy[:,0]>=crop[0]-.5-1e-8)
            assert np.all(xy[:,0]<=crop[2]-.5+1e-8)
            assert np.all(xy[:,1]>=crop[1]-.5-1e-8)
            assert np.all(xy[:,1]<=crop[3]-.5+1e-8)


def test_header_only_reader(tmp_path,monkeypatch):
    f=frame()
    fits.PrimaryHDU(np.ones(f.shape),header=f.wcs.to_header()).writeto(tmp_path/'real.fit')
    def no_pixels(*a,**k): raise AssertionError('pixel array read during geometry')
    monkeypatch.setattr(fits.PrimaryHDU,'_get_scaled_image_data',no_pixels)
    fs,rejected=g.read_frames(tmp_path)
    assert len(fs)==1 and not rejected
    assert g.run(fs,layouts=[(2,2)],halos=(0,))['frame_count']==1


def test_unsupported_projection_rejected():
    f=frame();f.wcs.wcs.ctype=['RA---SIN','DEC--SIN']
    with pytest.raises(ValueError,match='TAN'): g.make_canvas([f])


def test_legacy_cd_scale_bug_is_observable():
    w=frame().wcs
    assert np.mean(proj_plane_pixel_scales(w))==pytest.approx(.001)
    with pytest.warns(RuntimeWarning,match='cdelt'):
        assert gm._extract_pixel_scale_deg(w)==pytest.approx(1.)


def test_csv_order_metadata_not_sort_and_header_case(tmp_path):
    for name in ('a.fit','b.fit'): (tmp_path/name).touch()
    csv=tmp_path/'stack_plan.csv'
    csv.write_text('path,order,exposure,mount\nb.fit,99,20,EQ\na.fit,0,10,ALTZ\n')
    fs=gm.load_stack_plan(csv)
    assert [f.path.name for f in fs]==['b.fit','a.fit']
    assert [f.order for f in fs]==[99,0]
    csv.write_text('PATH,ORDER\na.fit,0\n')
    assert gm.load_stack_plan(csv)==[]


def request(images,**kw):
    args=dict(images=images,geometric_support=[np.ones(x.shape,bool) for x in images],
              normalization='none',weighting='none',rejection='none',combine='mean',taper='none')
    args.update(kw)
    return CanonicalStackRequest(**args)


def test_auto_reference_tie_is_input_index_not_frame_identity():
    ramp=np.arange(1024,dtype=np.float32).reshape(32,32)+100
    a,b=ramp,ramp*1.1+12
    normal=run_canonical_stack(request([a,b],normalization='linear_fit'))
    reverse=run_canonical_stack(request([b,a],normalization='linear_fit'))
    assert normal.provenance['reference']['index']==reverse.provenance['reference']['index']==0
    np.testing.assert_allclose(normal.science,a,atol=1e-4)
    np.testing.assert_allclose(reverse.science,b,atol=1e-4)
    assert not np.allclose(normal.science,reverse.science)


def test_median_chunk_fold_is_not_global_median():
    values=[0,1,2,100,101,102]
    images=[np.full((8,8),v,np.float32) for v in values]
    whole=run_canonical_stack(request(images,combine='median')).science
    def fold(groups):
        rs=[run_canonical_stack(request([images[i] for i in ids],combine='median')) for ids in groups]
        return sum(r.science*r.estimator_weight_sum for r in rs)/sum(r.estimator_weight_sum for r in rs)
    a=fold([[0,1,2],[3,4,5]])
    b=fold([[0,1,5],[2,3,4]])
    np.testing.assert_array_equal(whole,51.)
    np.testing.assert_array_equal(a,51.)
    np.testing.assert_array_equal(b,50.5)


def test_taper_needs_context_but_not_new_science():
    large=make_footprint_taper(np.ones((64,64),bool),8,0)
    direct=make_footprint_taper(np.ones((16,16),bool),8,0)
    padded=make_footprint_taper(np.ones((32,32),bool),8,0)[8:-8,8:-8]
    assert not np.array_equal(direct,large[24:40,24:40])
    np.testing.assert_array_equal(padded,large[24:40,24:40])


def test_legacy_alpha_is_not_reprojected_and_can_be_ignored(monkeypatch):
    f=gm.FrameInfo(Path('dummy'),wcs=frame().wcs,shape_hw=(8,8))
    tile=gm.GridTile(1,(0,4,0,4),copy.deepcopy(f.wcs))
    monkeypatch.setattr(gm,'_load_image_with_optional_alpha',lambda *a,**k:(np.ones((8,8,1),np.float32),np.zeros((8,8),np.float32)))
    monkeypatch.setattr(gm,'reproject_interp',lambda *a,**k:(np.ones((4,4)),np.ones((4,4))))
    _,support=gm._reproject_frame_to_tile(f,tile,(4,4))
    # Multiplication target(4,4) *= source(8,8) fails and is silently ignored.
    assert np.all(support==1)


def test_legacy_fallback_offset_is_not_baked_into_global_header():
    a=frame('a',angle=10,shape=(300,300));b=frame('b',ra=10.1,shape=(300,300))
    fs=[gm.FrameInfo(Path(f.key),wcs=f.wcs,shape_hw=f.shape) for f in (a,b)]
    w,shape,_,offset=gm._build_fallback_global_wcs(fs)
    assert offset!=(0,0)
    tile=gm._clone_tile_wcs(w,offset,shape)
    np.testing.assert_allclose(tile.wcs.crpix,w.wcs.crpix-np.array(offset))
    assert tile.pixel_to_world(0,0).separation(w.pixel_to_world(0,0)).arcsec>1
