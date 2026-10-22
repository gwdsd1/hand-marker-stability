import argparse
import json
import csv
from fractions import Fraction
import subprocess
import av
import cv2
from common import ROOT, np, assert_open, case_starts
from capture import Capture
from render import rasterize, overlay
from io_utils import save
from rgb_camera import RgbCamera,interpolate
from marker_overlay import annotate
from marker_model import MarkerHand
from common import torch,tensor


def distorted_render(vertices,faces,intrinsic,distortion,origin,shape):
    pixels=cv2.projectPoints(vertices,np.zeros(3),np.zeros(3),intrinsic,distortion)[0][:,0]
    virtual=vertices.copy()
    virtual[:,0]=(pixels[:,0]-intrinsic[0,2])/intrinsic[0,0]*vertices[:,2]
    virtual[:,1]=(pixels[:,1]-intrinsic[1,2])/intrinsic[1,1]*vertices[:,2]
    return rasterize(virtual,faces,np.column_stack([intrinsic,np.zeros(3)]),origin,shape)[0]


def encode(path,paths,times,speed=1.):
    container=av.open(str(path),mode='w')
    stream=container.add_stream('libx264',rate=Fraction(30)*Fraction(str(speed)))
    stream.width=1920
    stream.height=720
    stream.pix_fmt='yuv420p'
    stream.options=dict(crf='21',preset='fast')
    stream.time_base=Fraction(1,1000000)
    for image,time in zip(paths,times):
        frame=av.VideoFrame.from_ndarray(cv2.imread(str(image)),format='bgr24')
        frame.pts=round((time-times[0])/speed*1000000)
        frame.time_base=Fraction(1,1000000)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(path),'-f','null','-'],check=True)


def main():
    assert_open()
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit',required=True)
    parser.add_argument('--review-only',action='store_true')
    parser.add_argument('--inputs',default='inputs_v2')
    parser.add_argument('--starts',type=float,nargs='+')
    parser.add_argument('--discard-frame-images',action='store_true')
    parser.add_argument('--previous',default='previous_v2')
    parser.add_argument('--output-subdir',default='rgb_comparison')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--hands',nargs='+',choices=['left','right'],default=['left','right'])
    parser.add_argument('--track-crop',action='store_true')
    args=parser.parse_args()
    if not .1<=args.speed<=2:raise ValueError('Unsupported review playback speed')
    output=ROOT/args.fit/('review_frames' if args.review_only else args.output_subdir)
    output.mkdir(exist_ok=False)
    latest=[dict(np.load(ROOT/args.fit/f'{side}_fit.npz')) for side in args.hands]
    collections=[latest]
    observations=[dict(np.load(ROOT/args.inputs/f'{side}.npz')) for side in args.hands]
    capture=Capture()
    calibration=RgbCamera(capture)
    rows={int(row['rgb_index']):row for row in capture.index['rgb']}
    times=latest[0]['timestamps_seconds']
    query_times=np.array([calibration.time(rows[int(rgb)]) for rgb in latest[0]['rgb_indices']])
    if not np.array_equal(query_times,times):
        raise ValueError('Renderer requires the prepared C3D grid to match mapped RGB query times exactly')
    usable=(query_times>=times[0])&(query_times<=times[-1])
    for data in latest:
        data['vertices_world_m']=interpolate(data['vertices_world_m'],data['timestamps_seconds'],query_times[usable])
        data['rgb_indices']=data['rgb_indices'][usable]
        data['timestamps_seconds']=data['timestamps_seconds'][usable]
    for collection in collections:
        for side,data in zip(args.hands,collection):
            model=MarkerHand(np.zeros((14,3)),side)
            model.set_attachment(data)
            with torch.no_grad():
                surface,direction=model.surface(tensor(data['vertices_world_m']),data['marker_vertex_ids'])
            data['marker_surface_world_m']=surface.numpy()
    times=times[usable]
    query_times=query_times[usable]
    faces=np.concatenate([data['faces']+778*index for index,data in enumerate(latest)])
    receipts=[]
    for start in (args.starts if args.starts is not None else case_starts()):
        selected=np.flatnonzero((times>=start)&(times<start+4))
        if args.review_only:
            samples=[12.,12.3,12.6,12.9] if start==9 else [14.,14.1,14.2,14.3,14.4,14.6] if start==14 else [start+.3,start+1.5,start+3.0]
            selected=np.unique([selected[abs(times[selected]-time).argmin()] for time in samples])
        cameras={}
        projected=[]
        frame_bounds={}
        for frame in selected:
            row=rows[int(latest[0]['rgb_indices'][frame])]
            camera=calibration.matrix(row)
            cameras[int(frame)]=camera
            current=[]
            for data in collections:
                vertices=np.concatenate([hand['vertices_world_m'][frame] for hand in data])@camera[:3,:3].T+camera[:3,3]
                pixels=cv2.projectPoints(vertices,np.zeros(3),np.zeros(3),capture.intrinsics,capture.distortion)[0][:,0]
                projected.append(pixels)
                current.append(pixels)
            pixels=np.concatenate(current)
            if args.track_crop:
                observed = np.concatenate([observation['markers_world_m'][frame] for observation in observations])
                local_observed = observed @ camera[:3, :3].T + camera[:3, 3]
                measured_pixels = cv2.projectPoints(local_observed, np.zeros(3), np.zeros(3), capture.intrinsics, capture.distortion)[0][:, 0]
                pixels = np.concatenate([pixels, measured_pixels])
                visible = (pixels[:, 0] >= 0) & (pixels[:, 0] < 1920) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 1080)
                pixels = pixels[visible] if visible.any() else pixels
            frame_bounds[int(frame)]=(pixels.min(0),pixels.max(0))
        pixels=np.concatenate(projected)
        lower=np.maximum(np.floor(pixels.min(0)-50).astype(int),0)
        upper=np.minimum(np.ceil(pixels.max(0)+50).astype(int),[1920,1080])
        if args.track_crop:
            required = np.array([np.max(last - first) + 100 for first, last in frame_bounds.values()])
            extents = np.array([int(np.clip(np.ceil(required[max(0, index - 2):index + 3].max()), 240, 1080))
                for index in range(len(required))])
        paths=[]
        source_times=[]
        marker_rows=[]
        for ordinal,frame in enumerate(selected):
            rgb=int(latest[0]['rgb_indices'][frame])
            row=rows[rgb]
            camera=cameras[int(frame)]
            if args.track_crop:
                extent = int(extents[ordinal])
                first,last=frame_bounds[int(frame)]
                center=(first+last)/2
                lower=np.clip(np.rint(center-extent/2).astype(int),[0,0],np.array([1920,1080])-extent)
                upper=lower+extent
            for hand,(side,observation) in enumerate(zip(args.hands,observations)):
                observed=observation['markers_world_m'][frame]
                predicted=latest[hand]['markers_predicted_world_m'][frame]
                world=np.r_[observed,predicted]
                local=world@camera[:3,:3].T+camera[:3,3]
                pixels=cv2.projectPoints(local,np.zeros(3),np.zeros(3),capture.intrinsics,capture.distortion)[0][:,0]
                for marker in range(14):
                    marker_rows.append([rgb,float(query_times[frame]),side,marker,
                        int(observation['anatomical_to_raw_columns'][frame,marker]),
                        str(observation['anatomical_source_labels'][frame,marker]) if 'anatomical_source_labels' in observation else '',
                        int(observation['marker_provenance'][frame,marker]),*observed[marker].tolist(),
                        *pixels[marker].tolist(),*pixels[14+marker].tolist(),
                        float(np.linalg.norm(observed[marker]-predicted[marker])*1000)])
            image=capture.image(row)
            crop=image[lower[1]:upper[1],lower[0]:upper[0]]
            panels=[crop]
            for data in collections:
                vertices=np.concatenate([hand['vertices_world_m'][frame] for hand in data])@camera[:3,:3].T+camera[:3,3]
                depth=distorted_render(vertices,faces,capture.intrinsics,capture.distortion,lower,crop.shape[:2])
                panels.append(overlay(crop,depth,[220,200,40]))
            factor=min(960/crop.shape[1],600/crop.shape[0])
            size=(round(crop.shape[1]*factor),round(crop.shape[0]*factor))
            canvas=np.full((720,1920,3),22,np.uint8)
            for column,(panel,title) in enumerate(zip(panels,['RGB + MARKER PROJECTIONS','SELECTED: MARKER-ONLY POSE'])):
                display=cv2.resize(panel,size)
                scaled_intrinsic=capture.intrinsics.copy()
                scaled_intrinsic[:2]*=factor
                for hand,(side,observation) in enumerate(zip(args.hands,observations)):
                    prediction=None if column==0 else latest[hand]['markers_predicted_world_m'][frame]
                    annotate(display,side,observation['markers_world_m'][frame],observation['marker_provenance'][frame],
                        observation['anatomical_to_raw_columns'][frame],camera,scaled_intrinsic,capture.distortion,lower*factor,prediction,
                        observation['anatomical_source_labels'][frame] if 'anatomical_source_labels' in observation else None,
                        None if column==0 else latest[hand]['marker_surface_world_m'][frame])
                canvas[55:55+size[1],column*960:column*960+size[0]]=display
                cv2.putText(canvas,title,(column*960+15,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            cv2.putText(canvas,'C3D: YELLOW RING = RAW, ORANGE DIAMOND = INTERPOLATED; MAGENTA X = FIT; * = REMAPPED ID',(15,665),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1)
            cv2.putText(canvas,'BLUE DOT = MESH SKIN ATTACHMENT; BLUE LINE = SHARED NORMAL OFFSET',(15,642),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,235,235),1)
            speed_label='original speed' if args.speed==1 else f'{args.speed:g}x slow motion; no synthetic frames'
            cv2.putText(canvas,f'RGB {rgb}; C3D {query_times[frame]:.3f}s; {speed_label}; FRAME SYNC UNVERIFIED',(15,695),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1)
            if 'sample_reassociation_accepted' in observations[0] and observations[0]['sample_reassociation_accepted'][frame]:
                cv2.putText(canvas,'C3D WHOLE-HAND REASSOCIATION; * = INFERRED ID, @ = NATIVE UNLABELLED TRACK',(15,616),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,210,255),1)
            if 'sample_reassociation_quarantined' in observations[0] and observations[0]['sample_reassociation_quarantined'][frame]:
                cv2.putText(canvas,'UNCERTAIN GAP ID: FINGER OBSERVATIONS EXCLUDED; ? = MODEL PREDICTION ONLY',(15,616),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,130,255),1)
            path=output/f'clip{start:.1f}_RGB{rgb}.jpg'
            cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
            paths.append(path)
            source_times.append(capture.rgb_time(row))
            if ordinal%60==0:
                print('COMPARE_RENDER',args.fit,start,ordinal,len(selected),flush=True)
        if not args.review_only:
            path=output/f'comparison_{start:.1f}s.mp4'
            encode(path,paths,source_times,args.speed)
            video=cv2.VideoCapture(str(path))
            decoded=0
            while True:
                success,unused=video.read()
                if not success:
                    break
                decoded+=1
            video.release()
            receipts.append(dict(path=str(path),bytes=path.stat().st_size,source_frames=len(paths),decoded_frames=decoded,
                full_decode_passed=decoded==len(paths),source_span_seconds=source_times[-1]-source_times[0]+1/30,
                original_speed=args.speed==1,playback_speed=args.speed,
                intended_playback_span_seconds=(source_times[-1]-source_times[0]+1/30)/args.speed,
                historical_failure_review_window=start==23.5))
            if decoded!=len(paths):raise RuntimeError('Incomplete encoded video')
            if args.discard_frame_images:
                for image_path in paths:image_path.unlink()
        with open(output/f'marker_tracks_{start:.1f}s.csv','w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['rgb_index','c3d_time_s','hand','anatomical_column','raw_source_column_in_hand_14_or_minus1','raw_source_label','provenance_0_missing_1_native_2_interpolated',
                'world_x_m','world_y_m','world_z_m','rgb_u_px','rgb_v_px','predicted_u_px','predicted_v_px','fit_error_mm'])
            writer.writerows(marker_rows)
    save(output/'receipt.json',dict(clips=receipts,panels=['RGB + C3D marker projections','selected marker-only pose'],calibration=calibration.receipt(),
        candidate_fit=args.fit,rendered_hands=args.hands,
        crop_tracks_hand=args.track_crop,tracking_changes_only_equal_RGB_crop_not_pose=True,
        crop_uses_visible_projected_mesh_and_markers=args.track_crop,
        crop_also_includes_observed_markers=True,crop_extent_rule='Per-frame visible mesh+marker span with five-frame maximum and100px margin;240..1080px; identical in both panels',
        depth_used_by_new_pose=False,original_speed=args.speed==1,playback_speed=args.speed,camera_and_crop_identical=True,pose_visual_validation_pending=True,
        hand_marker_annotations=True,annotation_source='anatomically reassociated C3D, not RGB detections',
        marker_legend=dict(native='yellow ring',short_gap_interpolation='orange diamond',model_center='magenta cross',
            reassociated_id='asterisk'),marker_id_order=['IWR','OWR','IHAND','OHAND','THM3','THM6','IDX3','IDX6','MID0','MID6','RNG3','RNG6','PNK3','PNK6']))
    capture.close()


if __name__=='__main__':
    main()
