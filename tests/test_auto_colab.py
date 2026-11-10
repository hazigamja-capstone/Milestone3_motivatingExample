import ast
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
    def test_payload_unchanged_and_all_cells_parse(self):
        old=json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text())
        new=automatic_notebook(old)
        self.assertEqual(old['cells'][4],new['cells'][4])
        for cell in new['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))

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
