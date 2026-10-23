"""Protocol experiment adapter: immutable config, explicit simulation-only output."""
from pathlib import Path

from etx.api.types import Configuration
from etx.experiments import protocol


def run_protocol_experiment(config, *, output, executor=None):
    config = Configuration(config).to_dict()
    return protocol.run(config, Path(output), executor=executor)


def run_protocol_study(name, *, values=None, size=None):
    from etx.experiments.studies import run
    return Configuration(run(name, values=values, size=size))


def report_protocol_experiment(output):
    return protocol.report(Path(output))
