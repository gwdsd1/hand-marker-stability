import cv2
from common import np


NAMES=['IW','OW','IH','OH','T3','T6','I3','I6','M0','M6','R3','R6','P3','P6']


def annotate(image,side,positions,provenance,mapping,transform,intrinsic,distortion,origin,predicted=None,source_labels=None,surface=None):
    local=positions@transform[:3,:3].T+transform[:3,3]
    finite=np.isfinite(local).all(-1)&(local[:,2]>.05)&(provenance>0)
    pixels=np.full((len(local),2),np.nan)
    pixels[finite]=cv2.projectPoints(local[finite],np.zeros(3),np.zeros(3),intrinsic,distortion)[0][:,0]-origin
    if predicted is not None:
        prediction=predicted@transform[:3,:3].T+transform[:3,3]
        prediction_pixels=cv2.projectPoints(prediction,np.zeros(3),np.zeros(3),intrinsic,distortion)[0][:,0]-origin
    if surface is not None:
        skin=surface@transform[:3,:3].T+transform[:3,3]
        skin_pixels=cv2.projectPoints(skin,np.zeros(3),np.zeros(3),intrinsic,distortion)[0][:,0]-origin
    for index in np.flatnonzero(finite):
        point=tuple(np.rint(pixels[index]).astype(int))
        if not (0<=point[0]<image.shape[1] and 0<=point[1]<image.shape[0]):
            continue
        color=(0,220,255) if provenance[index]==1 else (0,135,255)
        if provenance[index]==1:
            cv2.circle(image,point,4,color,1,cv2.LINE_AA)
        else:
            diamond=np.array([[point[0],point[1]-5],[point[0]+5,point[1]],[point[0],point[1]+5],[point[0]-5,point[1]]],np.int32)
            cv2.polylines(image,[diamond],True,color,1,cv2.LINE_AA)
        label=side[0].upper()+':'+NAMES[index]+('*' if mapping[index]!=index else '')
        if source_labels is not None and source_labels[index].startswith('*'):
            label+='@'+source_labels[index][1:]
        offset=(5,-5) if index%2 else (5,11)
        location=(point[0]+offset[0],point[1]+offset[1])
        cv2.putText(image,label,location,cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(image,label,location,cv2.FONT_HERSHEY_SIMPLEX,.44,color,1,cv2.LINE_AA)
        if predicted is not None and np.isfinite(prediction_pixels[index]).all():
            fitted=tuple(np.rint(prediction_pixels[index]).astype(int))
            cv2.line(image,point,fitted,(150,100,150),1,cv2.LINE_AA)
            cv2.drawMarker(image,fitted,(255,60,240),cv2.MARKER_TILTED_CROSS,6,1,cv2.LINE_AA)
            if surface is not None and index>=2 and np.isfinite(skin_pixels[index]).all():
                anchor=tuple(np.rint(skin_pixels[index]).astype(int))
                cv2.line(image,anchor,fitted,(255,200,60),1,cv2.LINE_AA)
                cv2.circle(image,anchor,2,(255,100,30),-1,cv2.LINE_AA)
    if predicted is not None:
        for index in np.flatnonzero(~finite):
            if not np.isfinite(prediction_pixels[index]).all():
                continue
            point=tuple(np.rint(prediction_pixels[index]).astype(int))
            if 0<=point[0]<image.shape[1] and 0<=point[1]<image.shape[0]:
                cv2.drawMarker(image,point,(255,60,240),cv2.MARKER_TILTED_CROSS,6,1,cv2.LINE_AA)
                cv2.putText(image,side[0].upper()+':'+NAMES[index]+'?',(point[0]+4,point[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,60,240),1,cv2.LINE_AA)
    return image
