import ast
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import os
import signal
import time
import unittest
import zipfile

from paper_repro.colab_bundle import install_bundle
from paper_repro.colab_session import checkpoint_choice, run_logged


def sample_bundle(name='paper_repro/example.py', content=b'print(123)\n'):
    data = io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        z.writestr(name, content)
    return data.getvalue()


class BootstrapTests(unittest.TestCase):
    def test_source_restore_repeat_and_drive_source_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, study = Path(tmp)/'repo',Path(tmp)/'drive'
            payload = sample_bundle()
            sha = install_bundle(payload,repo,study)
            self.assertEqual(sha,hashlib.sha256(payload).hexdigest())
            self.assertEqual(install_bundle(payload,repo,study),sha)
            with self.assertRaisesRegex(ValueError,'different source'):
                install_bundle(sample_bundle(content=b'changed'),repo,study)
            (repo/'paper_repro/example.py').write_text('local edit')
            with self.assertRaisesRegex(ValueError,'modified'):
                install_bundle(payload,repo,study)

    def test_source_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                install_bundle(sample_bundle('../escape'),Path(tmp)/'repo',Path(tmp)/'study')
            self.assertFalse((Path(tmp)/'escape').exists())

    def test_small_gpu_default_and_frozen_checkpoint_choice(self):
        self.assertTrue(checkpoint_choice('auto',None,'ViT-L-14',{'total_gib':16}))
        self.assertFalse(checkpoint_choice('auto',None,'ViT-B-16',{'total_gib':16}))
        self.assertFalse(checkpoint_choice('auto',None,'ViT-L-14',{'total_gib':40}))
        self.assertTrue(checkpoint_choice('auto',{'grad_ckpt':True},'ViT-L-14',{'total_gib':80}))
        with self.assertRaises(ValueError):
            checkpoint_choice('off',{'grad_ckpt':True},'ViT-L-14')

    def test_logs_preserve_output_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'output.log'
            with self.assertRaises(subprocess.CalledProcessError) as failed:
                run_logged([sys.executable,'-c','print("intentional failure"); raise SystemExit(7)'],log)
            self.assertEqual(failed.exception.returncode,7)
            self.assertIn('intentional failure',log.read_text())

    def test_stop_cell_stops_scientific_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile=Path(tmp)/'worker.pid'
            worker = f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(60)"
            helper = ("import sys; from paper_repro.colab_session import run_logged; "
                      f"run_logged([sys.executable,'-c',{worker!r}],{str(Path(tmp)/'worker.log')!r})")
            wrapper = ("import sys; from paper_repro.colab_bundle import run_notebook_command; "
                       f"run_notebook_command([sys.executable,'-c',{helper!r}])")
            proc=subprocess.Popen([sys.executable,'-c',wrapper],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            worker_pid=None
            try:
                deadline=time.monotonic()+10
                while not pidfile.exists() and time.monotonic()<deadline:
                    time.sleep(.05)
                self.assertTrue(pidfile.exists())
                worker_pid=int(pidfile.read_text())
                proc.send_signal(signal.SIGINT)
                proc.communicate(timeout=15)
                self.assertNotEqual(proc.returncode,0)
                stat=Path(f'/proc/{worker_pid}/stat')
                self.assertTrue(not stat.exists() or stat.read_text().split()[2]=='Z')
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.communicate()
                if worker_pid:
                    try:
                        os.kill(worker_pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_generated_cells_use_stdlib_bootstrap_and_parse(self):
        from paper_repro.make_colab import cells
        generated=cells('B1',sample_bundle())
        for kind,text in generated:
            if kind=='code':
                ast.parse(text)
        text='\n'.join(value for _,value in generated)
        self.assertNotIn('git clone',text)
        self.assertNotIn('from paper_repro.run_plan import',text)
        self.assertIn('install_bundle(PAYLOAD',text)
        self.assertIn('colab_session.py',text)


if __name__=='__main__':
    unittest.main()
