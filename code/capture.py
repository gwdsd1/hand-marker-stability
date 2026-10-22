import csv
import json
import tarfile
from pathlib import Path
from common import ROOT,np
from scipy.spatial.transform import Rotation, Slerp
import cv2
import ezc3d

SOURCE = Path('/data1/DEV/egovicon_joint_captures/capture1019test2_20261019_095858_d74481')
SUFFIXES = ['IWR','OWR','IHAND','OHAND','THM3','THM6','IDX3','IDX6','MID0','MID6','RNG3','RNG6','PNK3','PNK6']


def camera(profile):
    return np.array([[profile['fx'],0,profile['cx']],[0,profile['fy'],profile['cy']],[0,0,1.]]),np.array([profile[key] for key in ['k1','k2','p1','p2','k3','k4','k5','k6']])


def undistort_pixels(pixels,intrinsics,distortion):
    target = (pixels-intrinsics[:2,2])/np.array([intrinsics[0,0],intrinsics[1,1]])
    points = target.copy()
    first,second,tangent_first,tangent_second,third,fourth,fifth,sixth = distortion
    for iteration in range(35):
        horizontal,vertical = points.T
        radius = horizontal**2+vertical**2
        numerator = 1+first*radius+second*radius**2+third*radius**3
        denominator = 1+fourth*radius+fifth*radius**2+sixth*radius**3
        radial = numerator/denominator
        derivative = ((first+2*second*radius+3*third*radius**2)*denominator-numerator*(fourth+2*fifth*radius+3*sixth*radius**2))/denominator**2
        projected = np.column_stack([horizontal*radial+2*tangent_first*horizontal*vertical+tangent_second*(radius+2*horizontal**2),
            vertical*radial+tangent_first*(radius+2*vertical**2)+2*tangent_second*horizontal*vertical])
        error = projected-target
        diagonal_first = radial+2*horizontal**2*derivative+2*tangent_first*vertical+6*tangent_second*horizontal
        cross = 2*horizontal*vertical*derivative+2*tangent_first*horizontal+2*tangent_second*vertical
        diagonal_second = radial+2*vertical**2*derivative+6*tangent_first*vertical+2*tangent_second*horizontal
        determinant = diagonal_first*diagonal_second-cross**2
        update = np.column_stack([(diagonal_second*error[:,0]-cross*error[:,1])/determinant,
            (diagonal_first*error[:,1]-cross*error[:,0])/determinant])
        points -= np.clip(update,-.1,.1)
        if np.max(np.abs(error))<1e-11:
            break
    if np.max(np.abs(error))>1e-7 or not np.isfinite(points).all():
        raise RuntimeError('Rational distortion inversion did not converge')
    return points


class Capture:
    def __init__(self,source=None):
        self.config = json.loads((ROOT/'capture_config.json').read_text())
        self.source = Path(source if source is not None else self.config['source'])
        self.initial_pose_path = self.source/self.config['initial_pose_npz']
        self.index = json.loads((self.source/'preview/playback_index.json').read_text())
        self.calibration = json.loads((self.source/'femto/rgb_depth_ir_calibration.json').read_text())
        self.intrinsics,self.distortion = camera(self.calibration['profiles']['rgb'])
        self.depth_intrinsics,self.depth_distortion = camera(self.calibration['profiles']['depth'])
        self.clock_start = self.index['mocap_start_rpc_midpoint_unix_seconds']
        self.initial_delta = self.index['motion_alignment_offset_seconds']
        native = ezc3d.c3d(str(self.source/self.config['native_c3d']))
        self.labels = native['parameters']['POINT']['LABELS']['value']
        self.fps = float(native['parameters']['POINT']['RATE']['value'][0])
        if native['parameters']['POINT']['UNITS']['value'] != ['mm']:
            raise ValueError('Expected documented C3D millimeters')
        raw = native['data']['points']
        self.points = raw[:3].transpose(2,1,0)*.001
        self.valid = (native['data']['meta_points']['residuals'][0].T>=0) & np.isfinite(self.points).all(-1) & (np.linalg.norm(self.points,axis=-1)>1e-6)
        self.points[~self.valid] = np.nan
        self.hand_columns = [self.labels.index(self.config['actor_prefix']+':'+side+name) for side in ['L','R'] for name in SUFFIXES]
        self.hand_labels = [self.labels[column].split(':')[1] for column in self.hand_columns]
        source_export = np.load(self.source/'mocap/markers_world_m.npz')
        time_map = dict(zip(source_export['frame_numbers'].tolist(),source_export['timestamps_seconds'].tolist()))
        samples = json.loads((self.source/'mocap/saved_markers.json').read_text())['samples']
        poses = []
        for sample in samples:
            frame = sample['data']['frame_number']
            for subject in sample['data']['subject_markers']:
                if subject['subject']==self.config['camera_body_prefix'] and subject['pose_valid'] and frame in time_map:
                    poses.append((time_map[frame],np.array(subject['pose_rotation']).reshape(3,3),np.array(subject['pose_xyz'])*.001))
        poses.sort(key=lambda row:row[0])
        self.pose_times = np.array([row[0] for row in poses])
        self.pose_rotations = np.array([row[1] for row in poses])
        self.pose_positions = np.array([row[2] for row in poses])
        self.slerp = Slerp(self.pose_times,Rotation.from_matrix(self.pose_rotations))
        self.body_columns = [self.labels.index(f'{self.config["camera_body_prefix"]}:M_{ordinal}') for ordinal in range(1,5)]
        body_local = []
        for time,rotation,translation in poses:
            frame = int(round(time*self.fps))
            body_local.append((self.points[frame,self.body_columns]-translation)@rotation)
        self.body_template = np.nanmedian(np.asarray(body_local),axis=0)
        self.interpolated_points = self.points.copy()
        self.interpolation_span = np.full(self.valid.shape,np.inf)
        self.interpolation_span[self.valid] = 0
        for column in self.hand_columns+self.body_columns:
            available = np.flatnonzero(self.valid[:,column])
            if len(available)<2:
                continue
            for lower,upper in zip(available[:-1],available[1:]):
                if 1<upper-lower<=21:
                    alpha = np.arange(1,upper-lower)/(upper-lower)
                    self.interpolated_points[lower+1:upper,column] = (1-alpha[:,None])*self.points[lower,column]+alpha[:,None]*self.points[upper,column]
                    self.interpolation_span[lower+1:upper,column] = (upper-lower)/self.fps
        self.handles,self.members,self.records = {},{},{}
        for folder in sorted((self.source/'femto/segments').iterdir()):
            archive_path = str(folder/'frames.tar')
            self.handles[archive_path] = open(archive_path,'rb')
            with tarfile.open(archive_path) as archive:
                self.members[archive_path] = {member.name:(member.offset_data,member.size) for member in archive if member.isfile()}
            with open(folder/'frames.csv') as stream:
                for row in csv.DictReader(stream):
                    self.records[int(row['rgb_index'])] = (archive_path,row)
        initial = json.loads(Path(self.config['initial_extrinsics_json']).read_text())
        self.external_calibration = initial
        self.initial_transform = np.linalg.inv(np.array(initial['transforms']['T_marker_from_rgb']))

    def read(self,row,key):
        if key!='rgb_file':
            raise RuntimeError('Depth/IR pixel reads are disabled in the RGB-beta / marker-pose pipeline')
        archive,record = self.records[int(row['rgb_index'])]
        offset,size = self.members[archive][record[key]]
        handle = self.handles[archive]
        handle.seek(offset)
        return handle.read(size)

    def image(self,row):
        return cv2.imdecode(np.frombuffer(self.read(row,'rgb_file'),np.uint8),cv2.IMREAD_COLOR)

    def depth_in_rgb(self,row):
        profile = self.calibration['profiles']['depth']
        shape = (profile['height'],profile['width'])
        if not hasattr(self,'depth_rays'):
            rows,columns = np.indices(shape)
            pixels = np.column_stack([columns.reshape(-1),rows.reshape(-1)]).astype(float)
            self.depth_rays = undistort_pixels(pixels,self.depth_intrinsics,self.depth_distortion)
        record = self.records[int(row['rgb_index'])][1]
        depths = np.frombuffer(self.read(row,'depth_file'),dtype='<u2').astype(float)*float(record['depth_scale_mm'])*.001
        valid = (depths>.1)&(depths<3)
        cloud = np.column_stack([self.depth_rays[valid]*depths[valid,None],depths[valid]])
        extrinsic = self.calibration['extrinsics']['depth_to_rgb']
        affine = np.asarray(extrinsic['rotation_sdk_rot']).reshape(3,3)
        translation = np.asarray(extrinsic['translation_mm'])*.001
        cloud = cloud@affine.T+translation
        pixels = cv2.projectPoints(cloud,np.zeros(3),np.zeros(3),self.intrinsics,self.distortion)[0][:,0]
        pixels = np.rint(pixels).astype(int)
        inside = (pixels[:,0]>=0)&(pixels[:,0]<1920)&(pixels[:,1]>=0)&(pixels[:,1]<1080)
        pixels,cloud = pixels[inside],cloud[inside]
        depth = np.full(1920*1080,np.inf,np.float32)
        np.minimum.at(depth,pixels[:,1]*1920+pixels[:,0],cloud[:,2])
        return depth.reshape(1080,1920),dict(rgb_depth_skew_us=int(record['rgb_device_timestamp_us'])-int(record['depth_device_timestamp_us']),
            factory_affine_orthogonality_error=float(np.linalg.norm(affine.T@affine-np.eye(3))),
            factory_affine_determinant=float(np.linalg.det(affine)),factory_transform_used_without_silent_rotation_projection=True)

    def rgb_time(self,row):
        return float(row['clock_unix_seconds']-self.clock_start)

    def query(self,time,hand_only=True):
        lower = int(np.floor(time*self.fps))
        if lower<0 or lower+1>=len(self.points):
            return None
        alpha = time*self.fps-lower
        columns = self.hand_columns if hand_only else np.arange(len(self.labels))
        first,second = self.interpolated_points[lower,columns],self.interpolated_points[lower+1,columns]
        if alpha<1e-6:
            return first.copy()
        return (1-alpha)*first+alpha*second

    def pose(self,time):
        points = self.query(time,hand_only=False)
        if points is None:
            return None
        body = points[self.body_columns]
        valid = np.isfinite(body).all(-1)
        if valid.sum()<3:
            return None
        source,target = self.body_template[valid],body[valid]
        first,unused,second = np.linalg.svd((source-source.mean(0)).T@(target-target.mean(0)))
        rotation = second.T@np.diag([1,1,np.linalg.det(second.T@first.T)])@first.T
        translation = target.mean(0)-rotation@source.mean(0)
        return rotation,translation

    def local(self,time):
        points,pose = self.query(time),self.pose(time)
        if points is None or pose is None:
            return None
        rotation,translation = pose
        return (points-translation)@rotation

    def project(self,points,transform):
        return cv2.projectPoints(points,cv2.Rodrigues(transform[:3,:3])[0],transform[:3,3],self.intrinsics,self.distortion)[0][:,0]

    def world_to_rgb(self,time,transform):
        rotation,translation = self.pose(time)
        inverse = np.eye(4)
        inverse[:3,:3] = rotation.T
        inverse[:3,3] = -rotation.T@translation
        return transform@inverse

    def close(self):
        for handle in self.handles.values():
            handle.close()
