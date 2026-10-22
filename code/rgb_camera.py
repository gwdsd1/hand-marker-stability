import json
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from common import ROOT, np
from io_utils import digest


JSON_PATH=Path('/data2/wangzhengwei/femto_sxt_rgb_extrinsics_candidate_20261020.json')
NPZ_PATH=JSON_PATH.with_suffix('.npz')


def interpolate(values,times,queries,rotations=False):
    queries=np.asarray(queries)
    if queries.min()<times[0] or queries.max()>times[-1]:
        raise ValueError('Requested time outside marker-only context')
    if rotations:
        shape=values.shape
        chains=values.reshape(len(values),-1,3)
        output=np.stack([Slerp(times,Rotation.from_rotvec(chains[:,joint]))(queries).as_rotvec() for joint in range(chains.shape[1])],axis=1)
        return output.reshape((len(queries),)+shape[1:])
    index=np.searchsorted(times,queries,side='right').clip(1,len(times)-1)
    fraction=(queries-times[index-1])/(times[index]-times[index-1])
    fraction=fraction.reshape((-1,)+(1,)*(values.ndim-1))
    return values[index-1]*(1-fraction)+values[index]*fraction


class RgbCamera:
    def __init__(self,capture,mode='recorded_frame_map'):
        self.capture=capture
        self.mode=mode
        correspondence=np.load(ROOT/'inputs_v2/left.npz')
        self.frame_times=dict(zip(correspondence['rgb_indices'].astype(int),correspondence['timestamps_seconds']))
        self.metadata=json.loads(JSON_PATH.read_text())
        numeric=dict(np.load(NPZ_PATH,allow_pickle=False))
        if self.metadata['translation_units']!='m' or str(numeric['translation_units'])!='m':
            raise ValueError('New extrinsics must have meter translations')
        if self.metadata['camera_marker_subject']!=capture.config['camera_body_prefix']:
            raise ValueError('Camera rigid body subject mismatch')
        self.transform=np.asarray(self.metadata['transforms']['T_rgb_from_marker'])
        inverse=np.asarray(self.metadata['transforms']['T_marker_from_rgb'])
        self.intrinsic=np.asarray(self.metadata['intrinsics']['K'])
        self.distortion=np.asarray(self.metadata['intrinsics']['D'])
        for key,value in [('T_rgb_from_marker',self.transform),('T_marker_from_rgb',inverse),('K',self.intrinsic),('D',self.distortion)]:
            if not np.allclose(numeric[key],value,atol=1e-12,rtol=0):
                raise ValueError('New JSON/NPZ calibration mismatch: '+key)
        if not np.allclose(self.transform@inverse,np.eye(4),atol=1e-10):
            raise ValueError('Calibration transforms are not inverse')
        if not np.allclose(self.transform[:3,:3].T@self.transform[:3,:3],np.eye(3),atol=1e-10) or np.linalg.det(self.transform[:3,:3])<.999999:
            raise ValueError('New extrinsic is not a proper rotation')
        capture.intrinsics=self.intrinsic
        capture.distortion=self.distortion

    def time(self,row):
        if self.mode=='recorded_frame_map':
            return float(self.frame_times[int(row['rgb_index'])])
        return self.capture.rgb_time(row)

    def matrix(self,row):
        return self.capture.world_to_rgb(self.time(row),self.transform)

    def receipt(self):
        return dict(json=str(JSON_PATH),npz=str(NPZ_PATH),json_sha256=digest(JSON_PATH),npz_sha256=digest(NPZ_PATH),
            translation_units='m',direction='sxt to RGB',dynamic_camera_composition='T_rgb_from_sxt @ inverse(T_world_from_sxt)',
            old_spatial_registration_used=False,new_temporal_offset_fitted=False,
            source_mapping_derived_from_legacy_motion_alignment=self.mode=='recorded_frame_map',
            time_correspondence=self.mode,
            review_only_conditional_correspondence=self.mode=='recorded_frame_map',
            hardware_sync_certified=False,source_status=self.metadata['status'])
