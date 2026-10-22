import argparse
import cv2
from common import ROOT,np,tensor,torch,assert_open
from capture import Capture
from rgb_camera import RgbCamera
from marker_model import MarkerHand
from measure_silhouettes import image_mask,model_mask
from io_utils import save,stats
from palm_view import palm_normals, facing_camera


def interval(mask,center,transverse,extent):
    offsets=np.arange(-extent,extent+.25,.25)
    points=center[None]+offsets[:,None]*transverse
    valid=(points[:,0]>=1)&(points[:,1]>=1)&(points[:,0]<mask.shape[1]-1)&(points[:,1]<mask.shape[0]-1)
    pixels=np.rint(points).astype(int)
    pixels[:,0]=np.clip(pixels[:,0],0,mask.shape[1]-1)
    pixels[:,1]=np.clip(pixels[:,1],0,mask.shape[0]-1)
    hit=(mask[pixels[:,1],pixels[:,0]]>0)&valid
    starts=np.flatnonzero(hit&~np.r_[False,hit[:-1]])
    ends=np.flatnonzero(hit&~np.r_[hit[1:],False])
    choices=[]
    for first,last in zip(starts,ends):
        width=offsets[last]-offsets[first]
        if not 4<=width<=40 or first==0 or last==len(offsets)-1:continue
        middle=(offsets[first]+offsets[last])/2
        choices.append(dict(center=float(middle),width=float(width),edges=[float(offsets[first]),float(offsets[last])]))
    return min(choices,key=lambda item:abs(item['center'])) if choices else None


def main():
    assert_open()
    parser=argparse.ArgumentParser()
    parser.add_argument('--fits',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--inputs',default='inputs_v2')
    parser.add_argument('--reference-records')
    args=parser.parse_args()
    allowed=None
    if args.reference_records:
        import json
        allowed={(row['rgb'],row['finger'],row['fraction']) for row in json.loads((ROOT/args.reference_records).read_text())['records']}
    output=ROOT/args.output;output.mkdir(exist_ok=False)
    capture=Capture();camera=RgbCamera(capture)
    rows={int(row['rgb_index']):row for row in capture.index['rgb']}
    reference=dict(np.load(ROOT/'baseline/left_fit.npz'))
    observation=dict(np.load(ROOT/args.inputs/'left.npz'))
    fits={name:dict(np.load(ROOT/name/'left_fit.npz')) for name in args.fits}
    model=MarkerHand(np.zeros((14,3)),'left');model.set_attachment(reference)
    directions=palm_normals(reference)
    times=reference['timestamps_seconds'];records=[]
    for start in [9.,14.,19.,23.5,28.5,30.8]:
        frames=np.unique([abs(times-(start+offset)).argmin() for offset in [.1,.4,.8,1.2,1.6,2.,2.5,3.,3.6]])
        for frame in frames:
            row=rows[int(reference['rgb_indices'][frame])];transform=camera.matrix(row)
            points=observation['markers_world_m'][frame]@transform[:3,:3].T+transform[:3,3]
            if not np.isfinite(points).all() or (points[:,2]<.1).any():continue
            pixels=cv2.projectPoints(points,np.zeros(3),np.zeros(3),camera.intrinsic,camera.distortion)[0][:,0]
            lower=np.maximum(np.floor(pixels.min(0)-65).astype(int),0)
            upper=np.minimum(np.ceil(pixels.max(0)+65).astype(int),[1920,1080])
            if (upper-lower<60).any():continue
            size=tuple((upper-lower).astype(int))
            image=capture.image(row)[lower[1]:upper[1],lower[0]:upper[0]]
            palm=(pixels[2:4].mean(0)-lower)
            actual=image_mask(image,palm)
            if actual is None:continue
            masks={name:model_mask(data['vertices_world_m'][frame],data['faces'],transform,camera,lower,size,1.) for name,data in fits.items()}
            toward=facing_camera(directions[frame],reference['joints_world_m'][frame,0],transform)
            view='dorsal' if toward>.35 else 'palm' if toward<-.35 else 'edge'
            panels=[image.copy() for unused in range(1+len(fits))]
            for finger,marker in [('index',6),('middle',8),('ring',10),('pinky',12)]:
                first,last=pixels[[marker,marker+1]]-lower
                direction=last-first;length=np.linalg.norm(direction)
                if length<17:continue
                direction/=length;transverse=np.array([-direction[1],direction[0]])
                for fraction in [.55,.7,.85]:
                    if allowed is not None and (int(row['rgb_index']),finger,fraction) not in allowed:continue
                    center=(1-fraction)*first+fraction*last
                    target=interval(actual,center,transverse,35)
                    if target is None or abs(target['center'])>target['width']*.55:continue
                    values={name:interval(mask,center,transverse,35) for name,mask in masks.items()}
                    if any(value is None for value in values.values()):continue
                    result={name:dict(signed_center_offset_px=value['center']-target['center'],
                        signed_screen_x_offset_px=float((value['center']-target['center'])*transverse[0]),
                        signed_screen_y_offset_px=float((value['center']-target['center'])*transverse[1]),
                        center_abs_offset_px=abs(value['center']-target['center']),
                        edge_abs_offset_px=float(np.mean(abs(np.array(value['edges'])-target['edges']))),
                        predicted_width_px=value['width']) for name,value in values.items()}
                    records.append(dict(clip_start=start,rgb=int(row['rgb_index']),time=float(times[frame]),finger=finger,
                        view=view,source_native=bool((observation['marker_provenance'][frame]==1).all()),
                        fraction=fraction,RGB_width_px=target['width'],transverse_screen=transverse.tolist(),results=result))
                    for column,panel in enumerate(panels):
                        for edge in target['edges']:
                            point=np.rint(center+edge*transverse).astype(int);cv2.circle(panel,tuple(point),2,(0,255,0),-1)
                        if column:
                            prediction=values[args.fits[column-1]]
                            for edge in prediction['edges']:
                                point=np.rint(center+edge*transverse).astype(int);cv2.circle(panel,tuple(point),2,(0,100,255),-1)
            canvas=np.full((image.shape[0]+40,image.shape[1]*len(panels),3),20,np.uint8)
            for ordinal,(name,panel) in enumerate(zip(['RGB']+args.fits,panels)):
                canvas[40:,ordinal*size[0]:(ordinal+1)*size[0]]=panel
                cv2.putText(canvas,name,(ordinal*size[0]+4,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imwrite(str(output/f'left_RGB{row["rgb_index"]}_{view}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
        print('FINGER_EVALUATION',start,flush=True)
    if allowed is not None and {(row['rgb'],row['finger'],row['fraction']) for row in records}!=allowed:
        raise RuntimeError('A frozen reference cross-section became unevaluable; do not silently drop it')
    summaries={}
    for group in ['all','dorsal','palm','edge']:
        selected=[record for record in records if record['finger'] in ['middle','ring','pinky'] and (group=='all' or record['view']==group)]
        summaries[group]={name:dict(center_abs_px=stats([row['results'][name]['center_abs_offset_px'] for row in selected]),
            signed_screen_x_px=stats([row['results'][name]['signed_screen_x_offset_px'] for row in selected]),
            edge_abs_px=stats([row['results'][name]['edge_abs_offset_px'] for row in selected]),
            frames=len(set(row['rgb'] for row in selected)),cross_sections=len(selected)) for name in fits}
    save(output/'result.json',dict(records=records,summary=summaries,
        score_is_RGB_segmentation_proxy_not_ground_truth=True,
        view_from_baseline_geometry_not_independent_annotation=True,
        view_definition='Fixed baseline wrist/index-MCP/pinky-MCP palm plane versus actual camera viewing ray; rest nail normals only establish dorsal sign',
        ambiguous_finger_overlap_and_blur_may_survive_gate=True,
        fitting_uses_no_RGB=True,selection_is_development_not_independent_test=True))
    capture.close()
    print('FINGER_SUMMARY',summaries,flush=True)


if __name__=='__main__':main()
