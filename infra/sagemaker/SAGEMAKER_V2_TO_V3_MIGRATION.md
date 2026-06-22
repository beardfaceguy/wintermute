# SageMaker Python SDK: V2 to V3 Migration Guide

Source: https://builder.aws.com/content/39mf1KfW5WUIjVf2mT4B00uwjcI/migrating-sagemaker-python-sdk-from-v2-to-v3-with-claude-code-a-technical-guide
Author: Ram Vittal (AWS Employee), Published: Feb 17, 2026

Wintermute is currently on **v2.257.3** (stable). V3 is Alpha as of 2026-06-21.
Check back Q3 2026 for Beta availability before migrating.

---

## Quick Reference: V2 → V3 API Mapping

| V2 | V3 | Notes |
|----|----|----|
| `Estimator` / `PyTorchEstimator` etc. | `ModelTrainer` | All estimators unified |
| `Model` / `HuggingFaceModel` etc. | `ModelBuilder` | All model classes unified |
| `Predictor` | `Endpoint` (returned by `deploy()`) | Different call pattern |
| `predictor.predict()` | `endpoint.invoke(body=..., content_type=...)` | Manual serialization |
| `model_data=` | `s3_model_data_url=` | Parameter renamed |
| `role=` | `role_arn=` | Parameter renamed |
| `sagemaker.get_execution_role()` | `from sagemaker.core.helper.session_helper import get_execution_role` | New import path |
| `sagemaker.Session()` | `from sagemaker.core.helper.session_helper import Session` | New import path |
| `image_uris.retrieve()` | `from sagemaker.core import image_uris` | New import path |

---

## Training: V2 → V3

### V2
```python
from sagemaker.pytorch import PyTorchEstimator

estimator = PyTorchEstimator(
    entry_point="train.py",
    role=role,
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    framework_version="2.0",
    py_version="py310",
    hyperparameters={"epochs": 10},
)
estimator.fit({"training": "s3://bucket/train"})
```

### V3
```python
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import InputData, Compute

train_input = InputData(
    channel_name="train",
    data_source="s3://bucket/train",
    content_type="application/x-recordio-protobuf"
)

trainer = ModelTrainer(
    training_image=container,
    role=role,                          # No custom TrainingJobName — auto-generated
    compute=Compute(
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        volume_size_in_gb=5
    ),
    hyperparameters={"epochs": 10}
)

trainer.train(input_data_config=[train_input], wait=True)

# Get auto-generated job name
job_name = trainer._latest_training_job.training_job_name
```

---

## Deployment: V2 → V3

### V2
```python
from sagemaker.model import Model

model = Model(image_uri=container, model_data="s3://...", role=role)
predictor = model.deploy(instance_type="ml.m5.xlarge", initial_instance_count=1)
result = predictor.predict(payload)
```

### V3
```python
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode

model_builder = ModelBuilder(
    s3_model_data_url="s3://...",       # NOT model_data
    role_arn=role,                       # NOT role
    image_uri=container,
    mode=Mode.SAGEMAKER_ENDPOINT
)

model = model_builder.build()
endpoint = model_builder.deploy(        # Call on model_builder, not model
    instance_type="ml.m5.xlarge",
    initial_instance_count=1,
    endpoint_name="my-endpoint",
    wait=True
)

# Manual serialization — no auto serializer/deserializer
result = endpoint.invoke(body=payload, content_type="application/json")
response_str = result.body.read().decode("utf-8")
```

---

## Hyperparameter Tuning: V2 → V3

```python
# V3
from sagemaker.train.tuner import HyperparameterTuner
from sagemaker.core.parameter import ContinuousParameter, IntegerParameter

hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.1),
    "epochs": IntegerParameter(5, 50),
}

base_trainer = ModelTrainer(
    training_image=container,
    role=role,
    compute=Compute(...),
    hyperparameters={"objective": "reg:linear"},
    sagemaker_session=sagemaker_session  # Pass session to TRAINER, not tuner
)

tuner = HyperparameterTuner(
    model_trainer=base_trainer,
    objective_metric_name="validation:rmse",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=6,
    max_parallel_jobs=2,
    strategy="Bayesian",
    objective_type="Minimize"
    # NOTE: No sagemaker_session parameter on tuner
)

tuner.tune(inputs=[train_input, validation_input], wait=True)
```

---

## Cleanup: V2 → V3

```python
# V3 — endpoint config must be deleted separately
endpoint_config_name = endpoint.endpoint_config_name
endpoint.delete()

from sagemaker.core.resources import EndpointConfig
endpoint_config = EndpointConfig.get(endpoint_config_name=endpoint_config_name)
endpoint_config.delete()
```

---

## Common Migration Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValidationError: Extra inputs are not permitted` for `compression` | V3 InputData doesn't accept `compression="None"` | Omit `compression` parameter |
| `ImportError: cannot import name 'ContinuousParameter' from 'sagemaker.train.configs'` | Wrong import path | Use `from sagemaker.core.parameter import ContinuousParameter` |
| `TypeError: HyperparameterTuner got unexpected keyword argument 'sagemaker_session'` | Session goes on trainer, not tuner | Pass `sagemaker_session` to `ModelTrainer`, not `HyperparameterTuner` |
| `ModuleNotFoundError: No module named 'sagemaker.model'` | Running from inside a directory named `sagemaker/` | Fix `sys.path` to exclude the script's own directory before imports |

---

## AWS Migration Tooling

The official migration skill for Claude Code lives at:
https://github.com/aws-samples/genai-ml-platform-examples/tree/main/migration/tools/smai-pysdk-migrator

Clone and add `.claude/skills/v2-to-v3-migration.md` to the target repo, then prompt:
```
Please use the v2-to-v3-migration skill to migrate <file> from V2 to V3.
```
