import ast
from common import ROOT, assert_open
from io_utils import save


def main():
    assert_open()
    entries = ['build_code_manifest', 'run_stability_pipeline', 'constrained_thumb', 'guard_stability',
        'stability_metrics', 'evaluate_stability_trials', 'evaluate_candidates', 'evaluate_finger_alignment',
        'measure_silhouettes', 'evaluate_side_views', 'gate_no_regression', 'select_stability',
        'render_comparison', 'review_videos', 'render_detail', 'create_research_copy',
        'test_thumb_constraints', 'verify_stability', 'stability_delivery', 'verify_delivery']
    available = {path.stem:path for path in ROOT.glob('*.py')}
    required = set(entries)
    pending = entries.copy()
    while pending:
        name = pending.pop()
        for node in ast.walk(ast.parse(available[name].read_text())):
            modules = [node.module] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            for module in modules:
                module = module.split('.')[0] if module else ''
                if module in available and module not in required:
                    required.add(module)
                    pending.append(module)
    save(ROOT / 'code_manifest.json', dict(python_files=sorted(name + '.py' for name in required),
        local_import_closure_verified=True, subprocess_entry_points=entries,
        raw_models_and_recorded_metadata_included=False,
        replay_inputs_required=['baseline', 'baseline_profile', 'inputs_v2', 'wrist_articulation',
            'evaluation_reference', 'evaluation_finger_records.json', 'VISUAL_REVIEW_RECORD.json']))


if __name__ == '__main__':
    main()
