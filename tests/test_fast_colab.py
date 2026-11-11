import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from paper_repro.colab_bundle import install_bundle
from paper_repro.make_fast_colab import fast_notebook
from paper_repro.compatibility import execution_sources

ROOT = Path(__file__).resolve().parents[1]


class FastNotebookTests(unittest.TestCase):
    def test_archives_are_separate_and_preserve_original_source_lock(self):
        old = json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text())
        new = fast_notebook(old)
        payloads = {}
        for cell in new['cells']:
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in ('PAYLOAD','FAST_PAYLOAD'):
                                payloads[target.id] = base64.b64decode(ast.literal_eval(node.value.args[0]))
        self.assertEqual(set(payloads), {'PAYLOAD','FAST_PAYLOAD'})
        self.assertTrue(''.join(new['cells'][4]['source']).startswith(''.join(old['cells'][4]['source'])))
        with tempfile.TemporaryDirectory() as tmp:
            study, original, backend = Path(tmp)/'study', Path(tmp)/'original', Path(tmp)/'backend'
            install_bundle(payloads['PAYLOAD'], original, study)
            old_bytes = {p:p.read_bytes() for p in original.rglob('*') if p.is_file()}
            backend_sha = hashlib.sha256(payloads['FAST_PAYLOAD']).hexdigest()
            install_bundle(payloads['FAST_PAYLOAD'], backend, study/'execution_backends'/backend_sha)
            install_bundle(payloads['FAST_PAYLOAD'], backend, study/'execution_backends'/backend_sha)
            self.assertTrue((backend/'paper_repro/compatibility_manifest.json').exists())
            execution_sources(backend, study)  # Validate the actually restored backend.
            self.assertEqual((study/'source_bundle.zip').read_bytes(), payloads['PAYLOAD'])
            for path, data in old_bytes.items():
                self.assertEqual(path.read_bytes(), data)
            with self.assertRaisesRegex(ValueError, 'different source'):
                install_bundle(payloads['FAST_PAYLOAD'], original, study)
        jobs = ''.join(new['cells'][8]['source'])
        self.assertNotIn("if (OUT/f'{combo}_plan.json').exists()", jobs)
        self.assertIn('FAST_SHA[:16]', jobs)

    def test_existing_plan_requires_backend_specific_smoke_once(self):
        import contextlib
        import os
        import sys
        from unittest.mock import patch
        nb = fast_notebook(json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text()), b'test backend')
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            study = Path(tmp)
            (study/'B1_plan.json').write_text('{}')
            namespace = dict(OUT=study, REPO=ROOT, VPY=Path('/fake/python'),
                             FAST_SHA='a'*64, BUNDLE_SHA='original', os=os, sys=sys)
            exec(''.join(nb['cells'][1]['source']), namespace)
            calls = []
            class Process:
                def __init__(self, command, **kwargs):
                    calls.append(command)
                    self.stdout = io.StringIO('smoke passed\n')
                def wait(self, **kwargs): return 0
                def poll(self): return 0
            with patch('subprocess.Popen', Process):
                exec(''.join(nb['cells'][8]['source']), namespace)
                self.assertEqual(sum(c[c.index('--phase')+1] == 'smoke' for c in calls), 4)
                calls.clear()
                exec(''.join(nb['cells'][8]['source']), namespace)
                self.assertFalse(any(c[c.index('--phase')+1] == 'smoke' for c in calls))
                calls.clear()
                namespace['FAST_SHA'] = 'b'*64
                exec(''.join(nb['cells'][8]['source']), namespace)
                self.assertEqual(sum(c[c.index('--phase')+1] == 'smoke' for c in calls), 4)
