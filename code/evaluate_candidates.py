import argparse
import json
from common import ROOT,np,assert_open
from io_utils import save,stats
from marker_model import MarkerHand


def main():
    assert_open()
    parser=argparse.ArgumentParser()
    parser.add_argument('--fits',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--inputs',default='inputs_v2')
    args=parser.parse_args()
    result={}
    for side in ['left','right']:
        observations=dict(np.load(ROOT/args.inputs/f'{side}.npz'))
        source=dict(np.load(ROOT/'baseline'/f'{side}_fit.npz'))
        model=MarkerHand(np.zeros((14,3)),side)
        model.set_shape_axes(source['beta'])
        hinge_indices,axes=model.hinges()
        axes=axes.numpy()
        collection={}
        for name in args.fits:
            fit=dict(np.load(ROOT/name/f'{side}_fit.npz'))
            error=np.linalg.norm(fit['markers_predicted_world_m']-observations['markers_world_m'],axis=-1)*1000
            native=observations['marker_provenance']==1
            times=fit['timestamps_seconds'];dt=np.diff(times)
            keep=(dt[1:]<.12)&(dt[:-1]<.12)
            velocity=np.diff(fit['joints_world_m'],axis=0)/dt[:,None,None]
            acceleration=2*np.diff(velocity,axis=0)/(dt[1:]+dt[:-1])[:,None,None]
            angles=(fit['pose'].reshape(-1,15,3)[:,hinge_indices]*axes[None]).sum(-1)
            groups={}
            for group,indices in dict(wrist=[0,1],palm=[2,3,8],proximal=[6,8,10,12],distal=[5,7,9,11,13],hand_surface=list(range(2,14))).items():
                groups[group]=stats(error[:,indices][native[:,indices]])
            blocks=[]
            for start in [9.,14.,19.,23.5,28.5,30.8]:
                selected=(times>=start)&(times<start+4)
                blocks.append(dict(start=start,all_target_mm=stats(error[selected]),
                    hand_surface_mm=stats(error[selected,2:]),wrist_mm=stats(error[selected,:2]),
                    distal_mm=stats(error[selected][:,[5,7,9,11,13]]),
                    compensatory_finger_frames=int(((angles[selected,::2]<0)&(angles[selected,1::2]>.4)).sum())))
            frozen_keys=['beta','marker_height_m','marker_vertex_ids','wrist_marker_local_offsets_m','marker_attachment_faces','marker_attachment_barycentric']
            collection[name]=dict(native_all_mm=stats(error[native]),native_regions_mm=groups,clips=blocks,
                over_12mm=int(((error>12)&native).sum()),
                negative_PIP_positive_DIP_count=int(((angles[:,::2]<0)&(angles[:,1::2]>.4)).sum()),
                coupled_angle_deviation_rad=stats(abs(angles[:,1::2]-.65*angles[:,::2])),
                diagnostic_angle_axes='same fixed actor-beta axes for all candidates, recomputed from saved pose',
                wrist_acceleration_m_s2=stats(np.linalg.norm(acceleration[keep,0],axis=-1)),
                all_joint_acceleration_m_s2=stats(np.linalg.norm(acceleration[keep],axis=-1)),
                fixed_shape_and_installation=all(np.array_equal(source[key],fit[key]) for key in frozen_keys),
                fixed_beta_and_skin_attachment=all(np.array_equal(source[key],fit[key]) for key in frozen_keys if key!='marker_height_m'))
        result[side]=collection
    save(ROOT/args.output,dict(hands=result,evaluation_inputs=args.inputs,same_common_evaluation_targets=True,
        targets_equal_legacy_inputs=args.inputs=='inputs_v2',
        changed_model_loss_does_not_change_evaluation_targets=True,
        anatomical_compensation_metric_is_a_diagnostic_not_pose_ground_truth=True))


if __name__=='__main__':
    main()
