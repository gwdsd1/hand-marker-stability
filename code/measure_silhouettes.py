import argparse
import json
import cv2
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from common import ROOT,np,assert_open,case_starts
from capture import Capture
from rgb_camera import RgbCamera
from io_utils import save,stats


def model_mask(vertices,faces,transform,camera,lower,size,factor):
    local=vertices@transform[:3,:3].T+transform[:3,3]
    pixels=cv2.projectPoints(local,np.zeros(3),np.zeros(3),camera.intrinsic,camera.distortion)[0][:,0]
    pixels=(pixels-lower)*factor
    output=np.zeros((size[1],size[0]),np.uint8)
    for face in faces[(local[faces,2]>.05).all(-1)]:
        cv2.fillConvexPoly(output,np.rint(pixels[face]).astype(np.int32),1)
    return output


def image_mask(image,palm):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    foreground=(((hsv[:,:,0]<28)|(hsv[:,:,0]>172))&(hsv[:,:,1]>24)&(hsv[:,:,1]<215)&(hsv[:,:,2]>40)).astype(np.uint8)
    foreground=cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    count,labels,statistics,centers=cv2.connectedComponentsWithStats(foreground,8)
    if count<2:
        return None
    scores=np.linalg.norm(centers[1:]-palm,axis=-1)+300/(np.maximum(statistics[1:,cv2.CC_STAT_AREA],1)**.5)
    selected=1+scores.argmin()
    mask=(labels==selected).astype(np.uint8)
    if mask.sum()<700:
        return None
    padded=np.pad(mask,1)
    flood=padded.copy()
    cv2.floodFill(flood,None,(0,0),1)
    padded|=1-flood
    return padded[1:-1,1:-1]


def edge(mask,valid):
    boundary=mask-cv2.erode(mask,np.ones((3,3),np.uint8))
    return (boundary>0)&valid


def score(mask,target,valid,factor):
    predicted=edge(mask,valid)
    actual=edge(target,valid)
    if predicted.sum()<25 or actual.sum()<25:
        return None
    distance=cv2.distanceTransform((~actual).astype(np.uint8),cv2.DIST_L2,5)
    reverse=cv2.distanceTransform((~predicted).astype(np.uint8),cv2.DIST_L2,5)
    values=np.r_[distance[predicted],reverse[actual]]/factor
    signed=(cv2.distanceTransform(1-target,cv2.DIST_L2,5)-cv2.distanceTransform(target,cv2.DIST_L2,5)).astype(np.float64)
    rows,columns=np.nonzero(predicted)
    stride=max(1,len(rows)//300)
    rows,columns=rows[::stride],columns[::stride]
    def residual(shift):
        return map_coordinates(signed,[rows+shift[1],columns+shift[0]],order=1,mode='nearest')
    apparent=least_squares(residual,np.zeros(2),bounds=(-30,30),loss='soft_l1',f_scale=3,diff_step=.1,max_nfev=35)
    union=((mask>0)|(target>0))&valid
    intersection=(mask>0)&(target>0)&valid
    return dict(contour_px=stats(values),silhouette_iou=float(intersection.sum()/max(union.sum(),1)),
        apparent_translation_px=(apparent.x/factor).tolist(),
        apparent_translation_length_px=float(np.linalg.norm(apparent.x)/factor),
        translation_is_diagnostic_only_not_applied=True)


def main():
    assert_open()
    parser=argparse.ArgumentParser()
    parser.add_argument('--fits',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference',default='baseline')
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(exist_ok=False)
    capture=Capture()
    camera=RgbCamera(capture)
    rows={int(row['rgb_index']):row for row in capture.index['rgb']}
    baseline={side:dict(np.load(ROOT/args.reference/f'{side}_fit.npz')) for side in ['left','right']}
    fits={name:{side:dict(np.load(ROOT/name/f'{side}_fit.npz')) for side in baseline} for name in args.fits}
    times=baseline['left']['timestamps_seconds']
    starts=[9.,14.,19.,23.5,28.5,30.8]
    records=[]
    for start in starts:
        frames=np.unique([abs(times-(start+offset)).argmin() for offset in [.1,.4,.8,1.2,1.6,2.,2.5,3.,3.6]])
        for frame in frames:
            rgb=int(baseline['left']['rgb_indices'][frame])
            row=rows[rgb]
            image=capture.image(row)
            transform=camera.matrix(row)
            for side,reference in baseline.items():
                markers=reference['markers_predicted_world_m'][frame]@transform[:3,:3].T+transform[:3,3]
                pixels=cv2.projectPoints(markers,np.zeros(3),np.zeros(3),camera.intrinsic,camera.distortion)[0][:,0]
                lower=np.maximum(np.floor(pixels.min(0)-65).astype(int),0)
                upper=np.minimum(np.ceil(pixels.max(0)+65).astype(int),[1920,1080])
                if (upper-lower<40).any():
                    continue
                factor=min(1.,380/max(upper-lower))
                size=tuple(np.maximum(2,np.rint((upper-lower)*factor).astype(int)))
                crop=cv2.resize(image[lower[1]:upper[1],lower[0]:upper[0]],size)
                joints=reference['joints_world_m'][frame][[0,4]]@transform[:3,:3].T+transform[:3,3]
                wrist,mcp=(cv2.projectPoints(joints,np.zeros(3),np.zeros(3),camera.intrinsic,camera.distortion)[0][:,0]-lower)*factor
                target=image_mask(crop,(wrist+mcp)/2)
                if target is None:
                    continue
                grid_y,grid_x=np.indices(target.shape)
                direction=(mcp-wrist)/max(np.linalg.norm(mcp-wrist),1)
                plane=(grid_x-wrist[0])*direction[0]+(grid_y-wrist[1])*direction[1]
                valid=plane>.06*np.linalg.norm(mcp-wrist)
                valid[:5]=False
                valid[-5:]=False
                valid[:,:5]=False
                valid[:,-5:]=False
                record=dict(clip_start=start,rgb=rgb,time=float(times[frame]),side=side,results={})
                panels=[crop.copy()]
                for name,collection in fits.items():
                    data=collection[side]
                    mask=model_mask(data['vertices_world_m'][frame],data['faces'],transform,camera,lower,size,factor)
                    metrics=score(mask,target,valid,factor)
                    if metrics is not None:
                        record['results'][name]=metrics
                    display=crop.copy()
                    display[mask>0]=(display[mask>0]*.7+np.array([220,200,40])*.3).astype(np.uint8)
                    display[edge(target,valid)]=[0,255,0]
                    display[edge(mask,valid)]=[0,120,255]
                    panels.append(display)
                height=max(panel.shape[0] for panel in panels)
                canvas=np.full((height+35,size[0]*len(panels),3),22,np.uint8)
                for column,(panel,title) in enumerate(zip(panels,['RGB']+args.fits)):
                    canvas[35:35+panel.shape[0],column*size[0]:(column+1)*size[0]]=panel
                    cv2.putText(canvas,title,(column*size[0]+3,23),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
                path=output/f'{side}_RGB{rgb}.jpg'
                cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,93])
                record['image']=path.name
                records.append(record)
        print('SILHOUETTE',start,flush=True)
    summary={}
    for start in starts:
        for side in baseline:
            selected=[record for record in records if record['clip_start']==start and record['side']==side]
            summary[f'{side}_{start:.1f}']={name:dict(
                frames=sum(name in record['results'] for record in selected),
                contour_median_px=stats([record['results'][name]['contour_px']['median'] for record in selected if name in record['results']]),
                contour_p95_px=stats([record['results'][name]['contour_px']['p95'] for record in selected if name in record['results']]),
                iou=stats([record['results'][name]['silhouette_iou'] for record in selected if name in record['results']]),
                diagnostic_translation_length_px=stats([record['results'][name]['apparent_translation_length_px'] for record in selected if name in record['results']])) for name in args.fits}
    save(output/'result.json',dict(records=records,summary=summary,calibration=camera.receipt(),
        segmentation='RGB color component and filled internal holes; fixed baseline ROI and wrist plane',
        ground_truth=False,diagnostic_translations_not_applied=True,rgb_or_depth_pose_optimization=False))
    capture.close()


if __name__=='__main__':
    main()
