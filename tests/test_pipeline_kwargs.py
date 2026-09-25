import unittest
from unittest.mock import MagicMock
import sys

# Mock third-party dependencies required when importing mimicmotion.pipelines.pipeline_mimicmotion
mock_modules = [
    'PIL', 'PIL.Image', 'einops', 'numpy', 'torch',
    'diffusers', 'diffusers.image_processor', 'diffusers.models',
    'diffusers.pipelines', 'diffusers.pipelines.pipeline_utils',
    'diffusers.pipelines.stable_diffusion', 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion',
    'diffusers.pipelines.stable_video_diffusion', 'diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion',
    'diffusers.schedulers', 'diffusers.utils', 'diffusers.utils.torch_utils',
    'transformers', 'mimicmotion.modules', 'mimicmotion.modules.pose_net'
]
for mod in mock_modules:
    sys.modules[mod] = MagicMock()

class BaseOutput:
    pass

class DiffusionPipeline:
    def register_modules(self, **kwargs):
        pass

sys.modules['diffusers.utils'].BaseOutput = BaseOutput
sys.modules['diffusers.pipelines.pipeline_utils'].DiffusionPipeline = DiffusionPipeline

from mimicmotion.pipelines.pipeline_mimicmotion import prepare_extra_step_kwargs, MimicMotionPipeline


class DummySchedulerWithEtaAndGenerator:
    def step(self, model_output, timestep, sample, eta=0.0, generator=None, return_dict=True):
        return (sample,)

class DummySchedulerWithoutEtaOrGenerator:
    def step(self, model_output, timestep, sample, return_dict=True):
        return (sample,)

class DummySchedulerWithEtaOnly:
    def step(self, model_output, timestep, sample, eta=0.0, return_dict=True):
        return (sample,)

class DummySchedulerWithGeneratorOnly:
    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        return (sample,)


class TestPrepareExtraStepKwargs(unittest.TestCase):
    def test_prepare_extra_step_kwargs_standalone_with_eta_and_generator(self):
        scheduler = DummySchedulerWithEtaAndGenerator()
        generator = MagicMock()
        eta = 0.0
        kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)

        self.assertEqual(kwargs, {"eta": 0.0, "generator": generator})

    def test_prepare_extra_step_kwargs_standalone_without_eta_or_generator(self):
        scheduler = DummySchedulerWithoutEtaOrGenerator()
        generator = MagicMock()
        eta = 0.0
        kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)

        self.assertEqual(kwargs, {})

    def test_prepare_extra_step_kwargs_standalone_with_eta_only(self):
        scheduler = DummySchedulerWithEtaOnly()
        generator = MagicMock()
        eta = 0.5
        kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)

        self.assertEqual(kwargs, {"eta": 0.5})

    def test_prepare_extra_step_kwargs_standalone_with_generator_only(self):
        scheduler = DummySchedulerWithGeneratorOnly()
        generator = MagicMock()
        eta = 0.5
        kwargs = prepare_extra_step_kwargs(scheduler, generator, eta)

        self.assertEqual(kwargs, {"generator": generator})

    def test_pipeline_method_delegation(self):
        pipeline = MimicMotionPipeline.__new__(MimicMotionPipeline)
        pipeline.scheduler = DummySchedulerWithEtaAndGenerator()
        generator = MagicMock()
        eta = 0.0
        kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

        self.assertEqual(kwargs, {"eta": 0.0, "generator": generator})


if __name__ == "__main__":
    unittest.main()
