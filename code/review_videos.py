import argparse
import cv2
from common import ROOT,np,assert_open
from io_utils import save,digest


def main():
    assert_open()
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit',default='selected_review')
    parser.add_argument('--output',default='decoded_video_review')
    parser.add_argument('--subdir',default='rgb_comparison')
    args=parser.parse_args()
    output=ROOT/args.output
    output.mkdir(exist_ok=False)
    records=[]
    for path in sorted((ROOT/args.fit/args.subdir).glob('*.mp4')):
        video=cv2.VideoCapture(str(path))
        count=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        picks=np.unique(np.rint(np.linspace(1,count-2,8)).astype(int))
        images=[]
        decoded=0
        while True:
            success,image=video.read()
            if not success:break
            if decoded in picks:
                frame_path=output/f'{path.stem}_frame{decoded:04d}.jpg'
                cv2.imwrite(str(frame_path),image,[cv2.IMWRITE_JPEG_QUALITY,95])
                images.append((decoded,cv2.resize(image,(1440,540))))
            decoded+=1
        video.release()
        for group in range(0,len(images),4):
            canvas=np.full((4*570,1440,3),20,np.uint8)
            selected=images[group:group+4]
            for ordinal,(frame,image) in enumerate(selected):
                cv2.putText(canvas,f'{path.name} decoded frame {frame}',(8,ordinal*570+23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                canvas[ordinal*570+30:ordinal*570+570]=image
            target=output/f'{path.stem}_sheet{group//4+1}.jpg'
            cv2.imwrite(str(target),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
            records.append(dict(video=path.name,sha256=digest(path),sheet=target.name,
                decoded_frame_indices=[int(frame) for frame,unused in selected],full_decoded_frames=decoded,
                visual_review_required=True))
    save(output/'receipt.json',dict(sheets=records,source='decoded final MP4, not pre-encode images',
        automatically_generated_contact_sheets_do_not_imply_visual_pass=True))


if __name__=='__main__':
    main()
