import pytest
from unittest.mock import MagicMock
from mimicmotion.utils.utils import prepare_extra_step_kwargs
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline


class DummySchedulerEtaAndGen:
    def step(self, sample, timestep, latents, eta=0.0, generator=None):
        pass


class DummySchedulerEtaOnly:
    def step(self, sample, timestep, latents, eta=0.0):
        pass


class DummySchedulerGenOnly:
    def step(self, sample, timestep, latents, generator=None):
        pass


class DummySchedulerNeither:
    def step(self, sample, timestep, latents):
        pass


def test_prepare_extra_step_kwargs_eta_and_generator():
    scheduler = DummySchedulerEtaAndGen()
    generator = "fake_generator"
    kwargs = prepare_extra_step_kwargs(scheduler, generator, eta=0.5)
    assert kwargs == {"eta": 0.5, "generator": "fake_generator"}


def test_prepare_extra_step_kwargs_eta_only():
    scheduler = DummySchedulerEtaOnly()
    generator = "fake_generator"
    kwargs = prepare_extra_step_kwargs(scheduler, generator, eta=0.5)
    assert kwargs == {"eta": 0.5}


def test_prepare_extra_step_kwargs_gen_only():
    scheduler = DummySchedulerGenOnly()
    generator = "fake_generator"
    kwargs = prepare_extra_step_kwargs(scheduler, generator, eta=0.5)
    assert kwargs == {"generator": "fake_generator"}


def test_prepare_extra_step_kwargs_neither():
    scheduler = DummySchedulerNeither()
    generator = "fake_generator"
    kwargs = prepare_extra_step_kwargs(scheduler, generator, eta=0.5)
    assert kwargs == {}


def test_pipeline_method_delegation():
    pipeline = MagicMock(spec=MimicMotionPipeline)
    pipeline.scheduler = DummySchedulerEtaAndGen()
    pipeline.prepare_extra_step_kwargs = MimicMotionPipeline.prepare_extra_step_kwargs.__get__(pipeline, MimicMotionPipeline)

    kwargs = pipeline.prepare_extra_step_kwargs("fake_generator", 0.7)
    assert kwargs == {"eta": 0.7, "generator": "fake_generator"}
