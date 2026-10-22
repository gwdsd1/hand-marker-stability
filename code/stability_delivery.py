import argparse
from datetime import datetime
import json
from zoneinfo import ZoneInfo
import zipfile
from common import ROOT, assert_open
from io_utils import save, digest


def package():
    names = json.loads((ROOT / 'code_manifest.json').read_text())['python_files']
    docs = ['README.md', 'FINAL_REPORT_ZH.md', 'ANNOTATION_FORMAT_ZH.md', 'VISUAL_REVIEW.md', 'VALIDATION_NOTES.md']
    code = ROOT / 'code_palm_thumb_stability_20261022.zip'
    with zipfile.ZipFile(code, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names + docs + ['capture_config.json', 'case_config.json', 'source.json', 'code_manifest.json']:
            archive.write(ROOT / name, name)
    directory = ROOT / 'selected_review/closeup_comparison'
    videos = sorted(directory.glob('*.mp4'))
    if len(videos) != 6:
        raise RuntimeError('Expected six final videos')
    video_zip = ROOT / 'videos_palm_thumb_stability_20261022.zip'
    with zipfile.ZipFile(video_zip, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in videos + sorted(directory.glob('*.csv')) + [directory / 'receipt.json']:
            archive.write(path, 'videos/' + path.name)
        for name in docs + ['selection.json', 'VISUAL_REVIEW_RECORD.json', 'RESEARCH_SUMMARY.json']:
            archive.write(ROOT / name, name)
    for path in [code, video_zip]:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('ZIP integrity failure')
    save(ROOT / 'delivery_receipt.json', dict(video_zip=dict(path=video_zip.name, bytes=video_zip.stat().st_size, sha256=digest(video_zip)),
        code_zip=dict(path=code.name, bytes=code.stat().st_size, sha256=digest(code)), python_files=len(names),
        total_MP4_bytes=sum(path.stat().st_size for path in videos),
        clips=[dict(path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=path.stat().st_size) for path in videos],
        final_panels=['original RGB with C3D provenance', 'latest left MANO marker-only pose'],
        playback_speed=.5, full_body_object_pipeline_updated=False,
        code_archive_alone_requires_existing_capture_and_metadata=True, model_assets_included=False))
    print('PACKAGED', video_zip.stat().st_size, code.stat().st_size, flush=True)


def freeze():
    for name in ['stability_verification.json', 'replay_final_verification.json']:
        if not json.loads((ROOT / name).read_text())['passed']:
            raise RuntimeError('Verification failed: ' + name)
    if not json.loads((ROOT / 'integrity_audit.json').read_text())['parent_and_dependencies_unchanged']:
        raise RuntimeError('Frozen parent changed')
    visual = json.loads((ROOT / 'VISUAL_REVIEW_RECORD.json').read_text())
    if not visual['no_obvious_added_deformation'] or len(visual['video_sha256']) != 6:
        raise RuntimeError('Visual review incomplete')
    parent = json.loads((ROOT.with_name('capture1019_observation_recovery_20261022') / 'finalization.json').read_text())
    files = sorted(path for path in ROOT.rglob('*') if path.is_file() and '.runtime' not in path.parts and path.name != 'finalization.json')
    save(ROOT / 'finalization.json', dict(status='complete_and_frozen_review_candidate',
        completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), selected_left='guarded_axes_v3', selected_right='baseline',
        files=len(files), sha256={str(path.relative_to(ROOT)):digest(path) for path in files},
        external_sha256=parent['external_sha256'], actual_final_visual_frames=48,
        final_decoded_frames=581, physical_optical_center_certified=False, production_default_changed=False,
        full_body_object_pipeline_updated=False,
        rerun_policy='Read-only verify_delivery.py; further research in a new sibling copy; no historical PID restoration'))
    print('FROZEN', len(files), 'dependencies', len(parent['external_sha256']), flush=True)


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['package', 'freeze'])
    args = parser.parse_args()
    package() if args.stage == 'package' else freeze()


if __name__ == '__main__':
    main()
