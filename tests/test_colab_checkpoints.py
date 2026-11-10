"""Colab restart regressions that also run with the pinned torch 2.2.1."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import torch

from paper_repro.checkpoint import mutable_model_state, save_state, load_state
from paper_repro.run_identity import mark_complete, completed_run, source_hashes
from paper_repro.run_plan import commands


class PartiallyFrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(2, 2, bias=False)
        self.frozen.requires_grad_(False)
        self.adaptive = torch.nn.Linear(2, 1)
        self.register_buffer('running', torch.tensor([.75]))

    def forward(self, x):
        return self.adaptive(self.frozen(x)) + self.running


def tta(model):
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=.01, momentum=.9)
    return types.SimpleNamespace(model=model, optimizer=optimizer, scaler=None,
                                 ema=.2, performed_updates=1)


class CompactCheckpointTests(unittest.TestCase):
    def test_only_mutable_parameters_and_buffers_are_persisted(self):
        model = PartiallyFrozenModel()
        self.assertEqual(set(mutable_model_state(model)),
                         {'adaptive.weight', 'adaptive.bias', 'running'})
        original = tta(model)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            save_state(path, original, 2, {'revision':'same'})
            state = torch.load(path, map_location='cpu', weights_only=False)
            self.assertEqual(state['version'], 3)
            self.assertNotIn('frozen.weight', state['model'])
            reconstructed = PartiallyFrozenModel()
            with torch.no_grad():
                reconstructed.frozen.weight.fill_(77)
                reconstructed.running.fill_(-2)
                reconstructed.adaptive.weight.fill_(-5)
                reconstructed.adaptive.bias.fill_(-3)
            frozen_before = reconstructed.frozen.weight.detach().clone()
            resumed = tta(reconstructed)
            self.assertEqual(load_state(path, resumed, {'revision':'same'}), 2)
            torch.testing.assert_close(reconstructed.frozen.weight, frozen_before)
            for key, expected in mutable_model_state(model).items():
                torch.testing.assert_close(reconstructed.state_dict()[key], expected)

    def test_progress_context_and_optimizer_update_survive_resume(self):
        torch.manual_seed(12)
        model = PartiallyFrozenModel()
        initial = copy.deepcopy(model.state_dict())
        original = tta(model)
        x = torch.tensor([[.1,.2], [.3,.4]])
        def step(adaptor):
            adaptor.optimizer.zero_grad()
            adaptor.model(x).square().mean().backward()
            adaptor.optimizer.step()
            with torch.no_grad():
                adaptor.model.running.add_(.1)
        step(original)
        context = dict(domain=1, batches=100, file='partial_dom01_b000100.npz')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            save_state(path, original, 1, {'revision':'same'}, progress=context)
            step(original)
            expected = copy.deepcopy(original.model.state_dict())
            reconstructed = PartiallyFrozenModel()
            reconstructed.load_state_dict(initial)
            resumed = tta(reconstructed)
            done, progress = load_state(path, resumed, {'revision':'same'}, with_progress=True)
            self.assertEqual(done, 1)
            self.assertEqual(progress, context)
            step(resumed)
            for name, value in resumed.model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_mismatched_trainability_or_legacy_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            save_state(path, tta(PartiallyFrozenModel()), 0, {'revision':'same'})
            model = PartiallyFrozenModel()
            model.frozen.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, 'trainable'):
                load_state(path, tta(model), {'revision':'same'})
            torch.save(dict(version=2, identity={'revision':'same'}), path)
            with self.assertRaisesRegex(ValueError, 'legacy'):
                load_state(path, tta(PartiallyFrozenModel()), {'revision':'same'})


class CompleteMarkerTests(unittest.TestCase):
    def fixture(self, root):
        identity = dict(method='eta', arch='ViT-B-16', role='paper', seed=1,
                        data_revision='a'*40, grad_ckpt=False, n_domains=3,
                        torch=torch.__version__, limit_classes=0, limit_batches=0,
                        checkpoint_batches=100, cli_opts=[],
                        source_sha256=source_hashes(Path(__file__).resolve().parents[1]))
        domains = ['gaussian_noise','shot_noise','impulse_noise']
        root.mkdir(parents=True)
        (root/'meta.json').write_text(json.dumps(dict(identity=identity, domains=domains)))
        (root/'state.pt').write_bytes(b'checkpoint')
        for index, domain in enumerate(domains):
            (root/f'dom{index:02d}_{domain}.npz').write_bytes(b'finished result')
        mark_complete(root, identity, domains)
        return identity

    def test_complete_marker_allows_skip_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'
            identity = self.fixture(root)
            self.assertTrue(completed_run(root, identity))
            with self.assertRaisesRegex(ValueError, 'different identity'):
                completed_run(root, dict(identity, seed=2))
            self.assertFalse(completed_run(Path(tmp)/'not_started', identity))

    def test_metadata_data_mutation_and_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case, filename in [('meta','meta.json'), ('data','dom00_gaussian_noise.npz'),
                                   ('state','state.pt')]:
                target = root/case
                identity = self.fixture(target)
                if case == 'state':
                    (target/filename).unlink()
                else:
                    (target/filename).write_bytes(b'altered')
                with self.assertRaises(ValueError):
                    completed_run(target, identity)

    def test_run_plan_skips_completed_seed_without_model_initialization(self):
        from paper_repro import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root/'eta_ViT-B-16_paper_s1')
            result = subprocess.run([sys.executable, run_plan.__file__, '--combo','B1',
                                     '--phase','paper','--seed','1', '--out',tmp,
                                     '--data_revision','a'*40], capture_output=True,text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('SKIP verified complete run', result.stdout)
            self.assertNotIn('RUN ', result.stdout)


class SessionSeedTests(unittest.TestCase):
    def test_only_registered_seed_can_be_selected(self):
        for role, seed in [('paper',1),('calibration',102),('confirmation',205)]:
            generated = commands('B1',role,Path('/tmp/study'),'a'*40,True,seed)
            self.assertEqual(len(generated),1)
            self.assertEqual(generated[0][-4:],['--role',role,'--seed',str(seed)])
            self.assertIn('--grad_ckpt', generated[0])
        for role, seed in [('paper',101),('calibration',201),('confirmation',101)]:
            with self.assertRaisesRegex(ValueError,'not registered'):
                commands('B1',role,Path('/tmp/study'),'a'*40,False,seed)

    def test_seed_on_noncompute_phase_fails_before_execution(self):
        from paper_repro import run_plan
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable,run_plan.__file__,'--combo','B1',
                                     '--phase','freeze','--seed','201','--out',tmp],
                                    capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('--seed is only valid',result.stderr)
            self.assertFalse(list(Path(tmp).iterdir()))


class ToyConfig(dict):
    """Only the configuration interface used by run_paper.main."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def clone(self):
        return copy.deepcopy(self)

    def defrost(self):
        pass

    def dump(self):
        return json.dumps(self, sort_keys=True)


class MidDomainResumeTests(unittest.TestCase):
    def test_actual_runner_replays_partial_domain_without_duplicate_predictions(self):
        """Run real runner/recording/checkpoint logic with a tiny CPU model registry.

        GPU kernels, upstream TTA algorithms, and remote images are deliberately
        outside this integration test; loss/backward/SGD and RNG are real torch.
        """
        import contextlib
        import io
        import os
        from unittest.mock import patch
        from paper_repro import run_paper

        class Interrupted(Exception):
            pass

        class ToyTTA:
            def __init__(self, cfg, model, num_classes):
                self.model = model
                self.optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
                self.initial = copy.deepcopy(model.state_dict())
                self.initial_optimizer = copy.deepcopy(self.optimizer.state_dict())
                self.scaler = None
                self.performed_updates = 0

            def reset(self):
                self.model.load_state_dict(self.initial)
                self.optimizer.load_state_dict(self.initial_optimizer)

            @torch.enable_grad()
            def __call__(self, x):
                outputs = self.model(x)
                loss = outputs.square().mean() * (1 + torch.rand(()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.performed_updates += 1
                return outputs

        def model_factory(cfg, count, device):
            model = torch.nn.Linear(2, count)
            with torch.no_grad():
                model.weight.copy_(torch.arange(count*2).reshape(count,2)/10000)
                model.bias.copy_(torch.arange(count)/10000)
            return model, lambda value:value

        def batch_factory(*args, **kwargs):
            ids = torch.arange(150)
            y = ids//50
            x = torch.stack([ids.float()/150,torch.ones(150)],dim=1)
            for i in range(0,150,64):
                yield x[i:i+64], y[i:i+64], ids[i:i+64]

        def load_cfg(_):
            torch.manual_seed(123)

        configuration = ToyConfig(
            MODEL=ToyConfig(ADAPTATION='eata',ARCH='ViT-L-14',USE_CLIP=True),
            CORRUPTION=ToyConfig(TYPE=['gaussian_noise'],SEVERITY=[5]),
            TEST=ToyConfig(BATCH_SIZE=64), OPTIM=ToyConfig(LR=.01),
            SETTING='correlated', RNG_SEED=1)
        modules = {
            'methods':types.ModuleType('methods'),
            'conf':types.SimpleNamespace(cfg=configuration, load_cfg_from_args=load_cfg),
            'models.model':types.SimpleNamespace(get_model=model_factory),
            'utils.registry':types.SimpleNamespace(ADAPTATION_REGISTRY=types.SimpleNamespace(get=lambda key:ToyTTA)),
        }
        original_to = torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == 'cuda':
                args = ('cpu',) + args[1:]
            return original_to(tensor,*args,**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def execute(out, interrupt=False):
                command = ['run_paper.py','--n_domains','1','--limit_classes','3',
                           '--checkpoint_batches','1','--data_revision','a'*40,'--out',str(out)]
                save = run_paper.save_state
                saved_progress = []
                def save_then_interrupt(*args, **kwargs):
                    save(*args, **kwargs)
                    if kwargs.get('progress') is not None:
                        saved_progress.append(kwargs['progress']['batches'])
                    if interrupt and kwargs.get('progress') is not None:
                        raise Interrupted()
                directory = Path.cwd()
                try:
                    with patch.dict(sys.modules,modules), patch.object(sys,'argv',command), \
                         patch.object(run_paper,'PAPER',tmp), patch.object(run_paper,'batches',batch_factory), \
                         patch.object(run_paper,'save_state',save_then_interrupt), \
                         patch.object(torch.Tensor,'to',cpu_to), \
                         patch.object(torch,'autocast',lambda *a,**kw:contextlib.nullcontext()), \
                         contextlib.redirect_stdout(io.StringIO()):
                        run_paper.main()
                finally:
                    os.chdir(directory)
                return saved_progress

            # The final 22-sample batch must never produce a full-batch partial
            # checkpoint, which would fail the resume sample-count invariant.
            self.assertEqual(execute(root/'continuous'), [1, 2])
            with self.assertRaises(Interrupted):
                execute(root/'resumed',interrupt=True)
            partial = root/'resumed'/'eta_ViT-L-14_paper_s1'
            self.assertTrue(list(partial.glob('partial_*.npz')))
            execute(root/'resumed')
            a = root/'continuous'/'eta_ViT-L-14_paper_s1'
            b = partial
            import numpy as np
            with np.load(a/'dom00_gaussian_noise.npz') as original, np.load(b/'dom00_gaussian_noise.npz') as resumed:
                self.assertEqual(len(resumed['sample_id']),150)
                self.assertEqual(len(np.unique(resumed['sample_id'])),150)
                for key in original.files:
                    np.testing.assert_array_equal(original[key],resumed[key])
            self.assertFalse(list(b.glob('partial_*.npz')))
            original_state = torch.load(a/'state.pt',map_location='cpu',weights_only=False)
            resumed_state = torch.load(b/'state.pt',map_location='cpu',weights_only=False)
            self.assertEqual(resumed_state['done'],1)
            self.assertIsNone(resumed_state['progress'])
            self.assertEqual(original_state['extra']['performed_updates'],resumed_state['extra']['performed_updates'])
            for key in original_state['model']:
                torch.testing.assert_close(original_state['model'][key],resumed_state['model'][key],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
