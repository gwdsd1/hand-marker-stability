import argparse
import cv2
from common import ROOT,np,torch,tensor,assert_open
from capture import Capture
from rgb_camera import RgbCamera
from marker_model import MarkerHand
from marker_overlay import annotate
from render_comparison import distorted_render
from render import overlay
from io_utils import save


def main():
    assert_open()
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--inputs',default='inputs_v2')
    parser.add_argument('--sides',nargs='+',default=['left','right'],choices=['left','right'])
    parser.add_argument('--times',nargs='+',type=float,default=[14.2,19.,28.8,30.,31.3])
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(exist_ok=False)
    capture=Capture();camera=RgbCamera(capture)
    rows={int(row['rgb_index']):row for row in capture.index['rgb']}
    records=[]
    for side in args.sides:
        source=dict(np.load(ROOT/args.fit/f'{side}_fit.npz'))
        baseline=dict(np.load(ROOT/'baseline'/f'{side}_fit.npz'))
        observation=dict(np.load(ROOT/args.inputs/f'{side}.npz'))
        models=[]
        for data in [baseline,source]:
            model=MarkerHand(np.zeros((14,3)),side);model.set_attachment(data);models.append(model)
        for time in args.times:
            frame=abs(source['timestamps_seconds']-time).argmin()
            row=rows[int(source['rgb_indices'][frame])]
            transform=camera.matrix(row)
            local=observation['markers_world_m'][frame]@transform[:3,:3].T+transform[:3,3]
            pixels=cv2.projectPoints(local,np.zeros(3),np.zeros(3),camera.intrinsic,camera.distortion)[0][:,0]
            mesh=source['vertices_world_m'][frame]@transform[:3,:3].T+transform[:3,3]
            mesh_pixels=cv2.projectPoints(mesh,np.zeros(3),np.zeros(3),camera.intrinsic,camera.distortion)[0][:,0]
            pixels=np.concatenate([pixels,mesh_pixels])
            visible=np.isfinite(pixels).all(-1)&(pixels[:,0]>=0)&(pixels[:,0]<1920)&(pixels[:,1]>=0)&(pixels[:,1]<1080)
            if not visible.any():continue
            pixels=pixels[visible]
            lower=np.maximum(np.floor(pixels.min(0)-55).astype(int),0)
            upper=np.minimum(np.ceil(pixels.max(0)+55).astype(int),[1920,1080])
            if (upper-lower<35).any():continue
            rgb=capture.image(row)[lower[1]:upper[1],lower[0]:upper[0]]
            panels=[rgb.copy()]
            for data in [baseline,source]:
                vertices=data['vertices_world_m'][frame]@transform[:3,:3].T+transform[:3,3]
                depth=distorted_render(vertices,data['faces'],camera.intrinsic,camera.distortion,lower,rgb.shape[:2])
                panels.append(overlay(rgb,depth,[220,200,40]))
            scale=min(700/rgb.shape[1],700/rgb.shape[0]);size=tuple(np.rint(np.array(rgb.shape[1::-1])*scale).astype(int))
            canvas=np.full((780,2100,3),22,np.uint8)
            intrinsic=camera.intrinsic.copy();intrinsic[:2]*=scale
            for column,panel in enumerate(panels):
                display=cv2.resize(panel,size)
                prediction=None;skin=None
                if column:
                    data=[baseline,source][column-1]
                    prediction=data['markers_predicted_world_m'][frame]
                    with torch.no_grad():skin,unused=models[column-1].surface(tensor(data['vertices_world_m'][frame:frame+1]),data['marker_vertex_ids'])
                    skin=skin.numpy()[0]
                annotate(display,side,observation['markers_world_m'][frame],observation['marker_provenance'][frame],
                    observation['anatomical_to_raw_columns'][frame],transform,intrinsic,camera.distortion,lower*scale,prediction,
                    observation['anatomical_source_labels'][frame],skin)
                canvas[45:45+size[1],column*700:column*700+size[0]]=display
                cv2.putText(canvas,['RGB + C3D','PREVIOUS','CANDIDATE'][column],(column*700+10,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
            cv2.putText(canvas,f'{side}; RGB {row["rgb_index"]}; C3D {source["timestamps_seconds"][frame]:.4f}s; fixed extrinsics and recorded conditional frame map',(10,760),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            path=output/f'{side}_RGB{row["rgb_index"]}.jpg';cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,96])
            records.append(dict(hand=side,time=float(source['timestamps_seconds'][frame]),rgb=int(row['rgb_index']),path=path.name))
    save(output/'receipt.json',dict(records=records,source_RGB_untouched=True,marker_annotations=True))
    capture.close()


if __name__=='__main__':
    main()
