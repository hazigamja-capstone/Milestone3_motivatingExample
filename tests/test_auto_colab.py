import ast
import base64
import hashlib
import zipfile
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from paper_repro.make_auto_colab import automatic_notebook

ROOT=Path(__file__).resolve().parents[1]

class AutoNotebookTests(unittest.TestCase):
    def test_payload_precision_fix_and_all_cells_parse(self):
        old=json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text())
        new=automatic_notebook(old)
        def payload(nb):
            tree = ast.parse(''.join(nb['cells'][4]['source']))
            node = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets))
            return base64.b64decode(ast.literal_eval(node.value.args[0]))
        original, fixed = payload(old), payload(new)
        self.assertIn(hashlib.sha256(fixed).hexdigest(), ''.join(new['cells'][4]['source']))
        with zipfile.ZipFile(io.BytesIO(original)) as a, zipfile.ZipFile(io.BytesIO(fixed)) as b:
            self.assertEqual(a.namelist(), b.namelist())
            for name in a.namelist():
                if name != 'paper_repro/run_paper.py':
                    self.assertEqual(a.read(name), b.read(name), name)
            self.assert_tta_autocast(b.read('paper_repro/run_paper.py').decode())
        self.assert_tta_autocast((ROOT/'paper_repro/run_paper.py').read_text())
        self.assertIn('motivating_exp3_v3_sar_fp16fix', ''.join(new['cells'][1]['source']))
        self.assertEqual(new, json.loads((ROOT/'colab_exp3_v3_auto.ipynb').read_text()))
        for cell in new['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))

    def assert_tta_autocast(self, source):
        # Execute the actual TTA call block. The fixture reproduces SAR's missing
        # internal autocast and rejects the original call outside that context.
        tree = ast.parse(source)
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.With)
                     and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                             and c.func.id == 'tta' for c in ast.walk(n))
                     and any(isinstance(i.context_expr, ast.Call)
                             and isinstance(i.context_expr.func, ast.Attribute)
                             and i.context_expr.func.attr == 'autocast' for i in n.items))
        active = []
        @contextlib.contextmanager
        def autocast(device, dtype):
            self.assertEqual((device, dtype), ('cuda', 'fp16'))
            active.append(True)
            try: yield
            finally: active.pop()
        class Logits:
            def float(self): return self
        def tta(x):
            self.assertTrue(active, 'FP32 input reaches FP16 SAR without autocast')
            return Logits()
        import types
        ns = dict(torch=types.SimpleNamespace(autocast=autocast, float16='fp16'), tta=tta, x=object())
        exec(compile(ast.Module(body=[block], type_ignores=[]), '<tta precision>', 'exec'), ns)
        self.assertIsInstance(ns['lt'], Logits)

    def test_order_complete_skip_and_failure_stop(self):
        nb=automatic_notebook(json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text()))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            ns=dict(OUT=Path(tmp),REPO=ROOT,VPY=Path('/fake/python'),BUNDLE_SHA='same',os=os,sys=sys)
            exec(''.join(nb['cells'][1]['source']),ns)
            self.assertEqual(len(ns['JOBS']),64)
            self.assertEqual(ns['JOBS'][:5],[('B1','smoke',None),('B1','paper',1),('B1','calibration',101),('B1','calibration',102),('B1','freeze',None)])
            calls=[]
            class Process:
                def __init__(self,cmd,**kwargs):
                    calls.append(cmd); self.stdout=io.StringIO('progress\n')
                def wait(self,**kwargs): return 0
                def poll(self): return 0
            with patch('subprocess.Popen',Process):
                exec(''.join(nb['cells'][8]['source']),ns)
            self.assertEqual(len(calls),64)
            for combo in ns['COMBOS']:
                (Path(tmp)/f'{combo}_plan.json').write_text('{}')
            calls.clear()
            with patch('subprocess.Popen',Process):
                exec(''.join(nb['cells'][8]['source']),ns)
            self.assertEqual(len(calls),60)  # four smoke jobs skipped
            self.assertFalse(any(c[c.index('--phase')+1]=='smoke' for c in calls))
            calls.clear()
            class Failing(Process):
                def wait(self,**kwargs): return 7
            with patch('subprocess.Popen',Failing), self.assertRaises(subprocess.CalledProcessError):
                exec(''.join(nb['cells'][8]['source']),ns)
            self.assertEqual(len(calls),1)

if __name__=='__main__': unittest.main()
