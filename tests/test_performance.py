import ast
import base64
from contextlib import closing
import copy
import io
import json
from pathlib import Path
import random
import shutil
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from paper_repro.performance import ordered_preprocess, prefetched_batches, reuse_fixed_text
from paper_repro.protocol import reorder_stream
from paper_repro.stream_cache import cached_source, cached_ordered_source
from paper_repro.compatibility import execution_sources, execution_plan_code, legacy_execution
from paper_repro.run_identity import prepare_run, source_hashes, completed_run, mark_complete
from paper_repro.checkpoint import save_state, load_state

ROOT = Path(__file__).resolve().parents[1]


def original_bundle():
    notebook = json.loads((ROOT/'colab_exp3_v3_ready.ipynb').read_text())
    for node in ast.walk(ast.parse(''.join(notebook['cells'][4]['source']))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in node.targets):
            return base64.b64decode(ast.literal_eval(node.value.args[0]))
    raise AssertionError('No original bundle')


def images(count=150):
    records = []
    for i in range(count):
        data = io.BytesIO()
        Image.new('RGB', (9, 7), (i % 256, 23, 45)).save(data, format='PNG')
        records.append((i//50, data.getvalue(), i))
    return records


def transform(image):
    return torch.from_numpy(np.asarray(image.resize((4, 4))).copy()).permute(2, 0, 1).float()/255


class InputPerformanceTests(unittest.TestCase):
    def test_parallelism_preserves_order_pixels_labels_ids_and_rng(self):
        records = images(32)
        active, peak = 0, 0
        lock = threading.Lock()
        def slow(image):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            time.sleep(.003 * (1 + int(np.asarray(image)[0, 0, 0]) % 3))
            tensor = transform(image)
            with lock:
                active -= 1
            return tensor
        py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        serial = list(ordered_preprocess(records, transform, workers=1))
        parallel = list(ordered_preprocess(records, slow, workers=4))
        self.assertGreater(peak, 1)
        for (a, y, i), (b, z, j) in zip(serial, parallel):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual((y, i), (z, j))
        self.assertEqual(random.getstate(), py_rng)
        np.testing.assert_equal(np.random.get_state(), np_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))

    def test_batch_boundaries_errors_and_early_close_release_workers(self):
        records = images()
        batches = list(prefetched_batches(records, transform, 64, workers=3))
        self.assertEqual([len(y) for _, y, _ in batches], [64, 64, 22])
        self.assertEqual(torch.cat([i for _, _, i in batches]).tolist(), list(range(150)))
        def fail(image):
            raise ValueError('decode/transform failed')
        with self.assertRaisesRegex(ValueError, 'decode/transform'):
            list(prefetched_batches(records, fail, 64, prefetch=1))
        with closing(prefetched_batches(records, transform, 1, prefetch=1)) as stream:
            next(stream)
        self.assertFalse(any(t.name == 'tta-input' or t.name.startswith('tta-decode')
                             for t in threading.enumerate()))

    def test_v1_cache_direct_order_equals_original_spooling_all_roles_and_seeds(self):
        records = images(300)
        with tempfile.TemporaryDirectory() as tmp:
            list(cached_source(lambda: iter(records), tmp, 'fixed', 6))
            for role in ('paper', 'calibration', 'confirmation'):
                for seed in (1, 101, 201):
                    expected = list(reorder_stream(iter(records), role, seed, 4, 6))
                    with patch('tempfile.TemporaryDirectory', side_effect=AssertionError('No class spooling')):
                        actual = list(cached_ordered_source(lambda: self.fail('Cache not reused'),
                                                           tmp, 'fixed', role, seed, 4, 6))
                    self.assertEqual(actual, expected)
            data = next(Path(tmp).glob('*/records.bin'))
            with data.open('r+b') as stream:
                stream.seek(-1, 2); stream.write(b'!')
            self.assertEqual(list(cached_ordered_source(lambda: iter(records), tmp, 'fixed', 'paper', 1, 4, 6)),
                             list(reorder_stream(records, 'paper', 1, 4, 6)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU required for pinned/nonblocking transfer')
    def test_pinned_nonblocking_transfer(self):
        for x, y, _ in prefetched_batches(images(70), transform, 64, pin_memory=True):
            self.assertTrue(x.is_pinned()); self.assertTrue(y.is_pinned())
            device_x, device_y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
            torch.testing.assert_close(device_x.cpu(), x, rtol=0, atol=0)
            torch.testing.assert_close(device_y.cpu(), y, rtol=0, atol=0)


class CompatibilityTests(unittest.TestCase):
    def test_exact_legacy_bundle_records_actual_backend_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            original = original_bundle()
            (study/'source_bundle.zip').write_bytes(original)
            expected = json.loads((ROOT/'paper_repro/compatibility_manifest.json').read_text())
            actual = execution_sources(ROOT, study)
            self.assertEqual(actual, {name: expected['legacy'][name] for name in actual})
            self.assertNotEqual(actual, source_hashes(ROOT))
            self.assertEqual(execution_plan_code(ROOT, study)['run_paper.py'],
                             expected['legacy']['paper_repro/run_paper.py'])
            records = list((study/'execution_backends').glob('*.json'))
            self.assertEqual(len(records), 1)
            provenance = json.loads(records[0].read_text())
            self.assertEqual(provenance['execution_source_sha256'], expected['optimized'])
            meta = dict(identity=dict(source_sha256=actual, seed=1), domains=['gaussian_noise'], gpu='original')
            run = study/'run'; run.mkdir()
            metadata = json.dumps(meta, indent=4).encode()
            (run/'meta.json').write_bytes(metadata)
            (run/'state.pt').write_bytes(b'preserved checkpoint')
            (run/'dom00_gaussian_noise.npz').write_bytes(b'preserved results')
            mark_complete(run, meta['identity'], meta['domains'])
            prepare_run(run, dict(meta, gpu='reconnected'))
            self.assertEqual((run/'meta.json').read_bytes(), metadata)
            self.assertTrue(completed_run(run, meta['identity']))
            with self.assertRaises(ValueError):
                prepare_run(run, dict(meta, identity=dict(meta['identity'], seed=2)))
            self.assertEqual((study/'source_bundle.zip').read_bytes(), original)
            records[0].write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                execution_sources(ROOT, study)

    def test_unknown_bundle_and_modified_backend_cannot_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)/'study'; study.mkdir()
            (study/'source_bundle.zip').write_bytes(b'unknown source version')
            self.assertEqual(execution_sources(ROOT, study), source_hashes(ROOT))
            (study/'source_bundle.zip').write_bytes(original_bundle())
            manifest = json.loads((ROOT/'paper_repro/compatibility_manifest.json').read_text())
            backend = Path(tmp)/'backend'
            for name in list(manifest['optimized'])+['paper_repro/compatibility_manifest.json']:
                target = backend/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, target)
            (backend/'paper_repro/performance.py').write_text('unverified modification')
            with self.assertRaisesRegex(ValueError, 'compatibility manifest'):
                legacy_execution(backend, study)

    def test_legacy_checkpoint_resume_exact_with_parallel_input(self):
        records = images()
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            (study/'source_bundle.zip').write_bytes(original_bundle())
            identity = dict(source_sha256=execution_sources(ROOT, study), seed=101, bs=64, lr=.01)
            torch.manual_seed(123)
            initial = torch.nn.Linear(48, 2).state_dict()
            def adaptor():
                model = torch.nn.Linear(48, 2); model.load_state_dict(initial)
                return types.SimpleNamespace(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.01, momentum=.9),
                                             scaler=None, performed_updates=0)
            def step(a, batch):
                x, y, ids = batch
                a.optimizer.zero_grad()
                logits = a.model(x.flatten(1))
                (logits.square().mean() * (1+torch.rand(()))).backward()
                a.optimizer.step(); a.performed_updates += 1
                return logits.detach().clone(), ids
            serial = list(prefetched_batches(records, transform, 64, workers=1))
            parallel = list(prefetched_batches(records, transform, 64, workers=4))
            a = adaptor(); start_rng = torch.get_rng_state()
            expected = [step(a, b) for b in serial]
            expected_state = copy.deepcopy(a.model.state_dict())
            resumed = adaptor(); torch.set_rng_state(start_rng)
            got = [step(resumed, parallel[0])]
            path = study/'state.pt'
            save_state(path, resumed, 0, identity, progress=dict(domain=0, batches=1))
            restored = adaptor()
            done, progress = load_state(path, restored, identity, with_progress=True)
            self.assertEqual((done, progress['batches']), (0, 1))
            got += [step(restored, b) for b in parallel[progress['batches']:]]
            for (x, ids), (y, other_ids) in zip(expected, got):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                self.assertTrue(torch.equal(ids, other_ids))
            for name, value in restored.model.state_dict().items():
                torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
            self.assertEqual(restored.performed_updates, a.performed_updates)


class TextReuseTests(unittest.TestCase):
    def test_pinned_upstream_constructor_features_logits_and_rng(self):
        import os
        path = Path(os.environ.get('PAPER_MODEL_SOURCE', ROOT/'third_party/test-time-adaptation/classification/models/model.py'))
        if not path.exists():
            self.skipTest('Set PAPER_MODEL_SOURCE to pinned upstream models/model.py')
        source = ast.parse(path.read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ZeroShotCLIP')
        module = types.ModuleType('pinned_test_model')
        module.__file__ = str(path)
        def tokenizer(texts):
            return torch.tensor([[len(t), sum(t.encode()) % 97] for t in texts])
        module.__dict__.update(torch=torch, nn=torch.nn, get_class_names=lambda dataset:['cat','dog'],
                               get_tokenizer=lambda arch:tokenizer, imagenet_templates=['a {}', 'the {}'],
                               logger=types.SimpleNamespace(info=lambda *a:None))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), module.__dict__)
        cfg = types.SimpleNamespace(CLIP=types.SimpleNamespace(PROMPT_MODE='ensemble', FREEZE_TEXT_ENCODER=True,
                                                               PROMPT_TEMPLATE=[]),
                                    CORRUPTION=types.SimpleNamespace(DATASET='imagenet_c'),
                                    MODEL=types.SimpleNamespace(ARCH='ViT-B-16', ADAPTATION='eata', WEIGHTS='openai'),
                                    dump=lambda:'same fixed configuration')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Linear(2, 2)
                self.transformer = torch.nn.Linear(2, 2)
                self.logit_scale = torch.nn.Parameter(torch.tensor(1.))
                self.calls = 0
            def encode_text(self, tokens):
                self.calls += 1
                return tokens.float() + .5
            def encode_image(self, images):
                return self.visual(images)
        reference = module.ZeroShotCLIP(cfg, Model(), 'cpu', torch.nn.Identity())
        state = torch.get_rng_state()
        ordinary = module.ZeroShotCLIP(cfg, Model(), 'cpu', torch.nn.Identity())
        after = torch.get_rng_state()
        torch.set_rng_state(state)
        original_class = module.ZeroShotCLIP
        with reuse_fixed_text(module, reference, cfg):
            fast = module.ZeroShotCLIP(cfg, Model(), 'cpu', torch.nn.Identity())
        self.assertIs(module.ZeroShotCLIP, original_class)
        self.assertEqual(ordinary.model.calls, 2)
        self.assertEqual(fast.model.calls, 0)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        torch.testing.assert_close(ordinary.text_features, fast.text_features, rtol=0, atol=0)
        self.assertNotEqual(fast.text_features.data_ptr(), reference.text_features.data_ptr())
        x = torch.tensor([[.1,.2], [.3,.4]])
        torch.testing.assert_close(ordinary(x), fast(x), rtol=0, atol=0)
        self.assertNotEqual(fast.model.visual.weight.data_ptr(), reference.model.visual.weight.data_ptr())


class PlanReuseTests(unittest.TestCase):
    def test_legacy_plan_completed_results_and_metadata_are_reused_unchanged(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            study = Path(tmp)
            (study/'source_bundle.zip').write_bytes(original_bundle())
            legacy = json.loads((ROOT/'paper_repro/compatibility_manifest.json').read_text())['legacy']
            plan = dict(version=3, combo='B1', data_revision='a'*40, grad_ckpt=False, n_domains=3,
                        calibration_seeds=[101,102], confirmation_seeds=list(range(201,211)),
                        code={n:legacy['paper_repro/'+n] for n in execution_plan_code(ROOT, study)},
                        prepare_inc_sha256=legacy['prepare_inc.py'])
            (study/'B1_plan.json').write_text(json.dumps(plan, indent=4))
            from test_colab_checkpoints import CompleteMarkerTests
            run = study/'eta_ViT-B-16_paper_s1'
            identity = CompleteMarkerTests().fixture(run)
            identity['source_sha256'] = execution_sources(ROOT, study)
            meta = json.loads((run/'meta.json').read_text()); meta['identity'] = identity
            (run/'meta.json').write_text(json.dumps(meta, indent=3))
            mark_complete(run, identity, meta['domains'])
            preserved = {p:p.read_bytes() for p in study.rglob('*') if p.is_file()}
            result = subprocess.run([sys.executable, ROOT/'paper_repro/run_plan.py','--combo','B1',
                                     '--phase','paper','--seed','1','--out',study],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('SKIP verified complete run', result.stdout)
            for path, data in preserved.items():
                self.assertEqual(path.read_bytes(), data, str(path))


class ClipTransformIntegrationTests(unittest.TestCase):
    def test_actual_eval_transform_batch_tensors_match_serial_and_reject_random(self):
        import importlib.util
        if importlib.util.find_spec('open_clip') is None:
            self.skipTest('open_clip needed for actual deterministic eval transform integration')
        from open_clip.transform import image_transform
        import torchvision.transforms as transforms
        from paper_repro.performance import deterministic_clip_transform
        from paper_repro import run_paper
        preprocess = image_transform(224, is_train=False)
        preprocess.transforms = preprocess.transforms[:-1]  # pinned upstream moves Normalize into the model
        self.assertTrue(deterministic_clip_transform(preprocess))
        stochastic = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        self.assertFalse(deterministic_clip_transform(stochastic))
        records = images(7)
        expected = torch.stack([preprocess(Image.open(io.BytesIO(encoded)).convert('RGB'))
                                for _, encoded, _ in records])
        with patch.object(run_paper, 'iter_split', return_value=iter(records)), \
                patch('torch.cuda.is_available', return_value=False):
            actual = list(run_paper.batches('gaussian_noise', 5, 1, preprocess, 64))
        self.assertEqual(len(actual), 1)
        torch.testing.assert_close(actual[0][0], expected, rtol=0, atol=0)
        self.assertEqual(actual[0][2].tolist(), list(range(7)))
