import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

from paper_repro import analyze_paper as analysis
from paper_repro.checkpoint import save_state, load_state
from paper_repro.run_identity import prepare_run


class AnalysisTests(unittest.TestCase):
    def fixture(self, root, seed=2, method="eta", n=50000):
        root.mkdir()
        y = np.arange(n) % 1000
        np.savez(root / "dom00_gaussian_noise.npz", y=y, pred_t=(y+1)%1000,
                 pred_z=y, ent_t=np.ones(n), maxp_t=np.full(n, .5),
                 kl_tz=np.ones(n), rank_z_at_t=np.ones(n))
        meta = dict(seed=seed, method=method, arch="ViT-B-16", setting="correlated",
                    domains=["gaussian_noise"], severity=5, bs=64, lr=.00025,
                    paper_commit="abc", torch="test", cfg=f"RNG_SEED: {seed}\nTEST: 1")
        (root / "meta.json").write_text(json.dumps(meta))
        return str(root)

    def test_duplicate_paths_do_not_create_independent_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.fixture(Path(tmp)/"s2")
            self.assertEqual(len(analysis.validate_runs([p, p, p])), 1)
            result = subprocess.run([sys.executable, analysis.__file__, p, p, p],
                                    capture_output=True, text=True, check=True)
            self.assertIn("판정 불가", result.stdout)
            self.assertNotIn("→ 통과", result.stdout)

    def test_duplicate_seed_and_mixed_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.fixture(Path(tmp)/"a")
            q = self.fixture(Path(tmp)/"b")
            with self.assertRaisesRegex(ValueError, "Duplicate seed"):
                analysis.validate_runs([p,q])
            q = self.fixture(Path(tmp)/"c", seed=3, method="sar")
            with self.assertRaisesRegex(ValueError, "Mixed"):
                analysis.validate_runs([p,q])

    def test_three_distinct_confirmation_seeds_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [self.fixture(Path(tmp)/f"s{s}", seed=s) for s in (2,3,4)]
            result = subprocess.run([sys.executable, analysis.__file__, *paths],
                                    capture_output=True, text=True, check=True)
            self.assertIn("3 / 3 실행  → 통과", result.stdout)

    def test_partial_and_smoke_runs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.fixture(Path(tmp)/"partial", n=1536)
            with self.assertRaisesRegex(ValueError, "50000"):
                analysis.validate_runs([p])
            q = Path(self.fixture(Path(tmp)/"smoke"))
            meta = json.loads((q/"meta.json").read_text()); meta["limit_classes"] = 3
            (q/"meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "smoke"):
                analysis.validate_runs([str(q)])

    def test_misses_count_in_summary(self):
        self.assertEqual(analysis.alarm_summary([0,None,None]), (None,1,3))
        self.assertEqual(analysis.alarm_summary([-2,0,None]), (0,2,3))
        self.assertEqual(analysis.alarm_summary([0,None]), (None,1,2))
        self.assertEqual(analysis.alarm_summary([None,None]), (None,0,2))

    def test_nan_and_noncontiguous_domains_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(self.fixture(Path(tmp)/"nan"))
            f = p/"dom00_gaussian_noise.npz"
            with np.load(f) as z:
                arrays = dict(z)
            arrays["kl_tz"][0] = np.nan
            np.savez(f, **arrays)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                analysis.validate_runs([str(p)])
            q = Path(self.fixture(Path(tmp)/"gap"))
            (q/"dom00_gaussian_noise.npz").rename(q/"dom01_gaussian_noise.npz")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                analysis.validate_runs([str(q)])

    def test_event_boundaries_and_false_alarm_clusters(self):
        row = dict(dom=0, mode_t=.1, mode_y=.1, mode_lab=1., mode_z=.1)
        rows = [dict(row, delta=x) for x in (0,-5,-6,-5,0,-5,-5)]
        events = analysis.find_events(rows)
        self.assertEqual([(e['start'],e['end']) for e in events],[(1,3)])
        self.assertEqual(analysis.first_alarm([False,False,True,True,False,False,False],events[0]),1)
        self.assertEqual(analysis.false_alarms(rows,[True]*7),(1.,2))


class IdentityTests(unittest.TestCase):
    def test_reconnect_and_rejected_extension_preserve_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = dict(identity={"lr":.01}, domains=["a"])
            prepare_run(tmp,meta)
            prepare_run(tmp,meta)
            with self.assertRaisesRegex(ValueError, "immutable"):
                prepare_run(tmp,dict(meta,domains=["a","b"]))
            before = (Path(tmp)/"meta.json").read_bytes()
            with self.assertRaises(ValueError):
                prepare_run(tmp,dict(identity={"lr":.02},domains=["a","b"]))
            self.assertEqual((Path(tmp)/"meta.json").read_bytes(),before)
            with self.assertRaises(ValueError):
                prepare_run(tmp,dict(meta,domains=["b","a"]))

    def test_legacy_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"state.pt").touch()
            with self.assertRaises(ValueError):
                prepare_run(tmp,dict(identity={},domains=["a"]))


def upstream_sam():
    # Use the exact pinned upstream SAM implementation, without importing its GPU dependencies.
    path = Path(os.environ.get('PAPER_SAR_SOURCE', '/nonexistent/pinned-sar.py'))
    if not path.exists():
        return None
    import ast
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SAM')
    namespace = {'torch':torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'),namespace)
    return namespace['SAM']


class SerializableScaler:
    """CPU scale/unscale fixture compatible with pinned torch 2.2.1.

    Exercises checkpointed scale/update state and real optimizer updates without
    claiming to test CUDA overflow detection or using post-2.2 CPU AMP APIs.
    """
    def __init__(self):
        self.factor = 8.0
        self.updates = 0

    def scale(self, loss):
        return loss * self.factor

    def step(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self.factor)
        optimizer.step()

    def update(self):
        self.updates += 1
        if self.updates % 2 == 0:
            self.factor *= 2

    def state_dict(self):
        return dict(factor=self.factor, updates=self.updates)

    def load_state_dict(self, state):
        self.factor = state['factor']
        self.updates = state['updates']


class CheckpointTests(unittest.TestCase):
    def model(self, sam=None):
        model = torch.nn.Linear(2,1)
        opt = sam(model.parameters(),torch.optim.SGD,lr=.01,momentum=.9) if sam else torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        return types.SimpleNamespace(model=model,optimizer=opt,scaler=SerializableScaler(),ema=.4,performed_updates=0)

    def step(self, tta, sam=False):
        x = torch.randn(4,2)
        tta.optimizer.zero_grad()
        if sam:
            tta.model(x).square().mean().backward()
            tta.optimizer.first_step(zero_grad=True)
            tta.model(x).square().mean().backward()
            tta.optimizer.second_step(zero_grad=True)
        else:
            loss = tta.model(x).square().mean()
            tta.scaler.scale(loss).backward()
            tta.scaler.step(tta.optimizer)
            tta.scaler.update()
        tta.performed_updates += 1
        return random.random(), np.random.rand()

    def check_resume(self, sam=None):
        torch.manual_seed(10); random.seed(10); np.random.seed(10)
        original = self.model(sam)
        self.step(original, bool(sam))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'state.pt'
            save_state(p,original,1,{'test':1})
            expected_rng = self.step(original, bool(sam))
            expected = copy.deepcopy(original.model.state_dict())
            resumed = self.model(sam)
            self.assertEqual(load_state(p,resumed,{'test':1}),1)
            self.assertEqual(self.step(resumed,bool(sam)),expected_rng)
            for k,v in resumed.model.state_dict().items():
                torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertEqual(original.scaler.state_dict(),resumed.scaler.state_dict())
            self.assertEqual(original.performed_updates,resumed.performed_updates)
            with self.assertRaises(ValueError):
                load_state(p,resumed,{'test':2})

    def test_scaler_optimizer_and_rng_resume(self):
        self.check_resume()

    def test_upstream_sam_momentum_resume(self):
        sam = upstream_sam()
        if sam is None:
            self.skipTest('Set PAPER_SAR_SOURCE to the pinned upstream sar.py for this integration test')
        self.check_resume(sam)


if __name__ == '__main__':
    unittest.main()
