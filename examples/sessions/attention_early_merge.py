"""C6 probe only. Single GPU, fixed 40-minute cap, no provisioning.

After local prerequisites pass and the owner authorizes GPU use:
 C6_WORK=/absolute/new/session C6_MODEL=/absolute/packed/qwen \
 python -m etx.tools.gpu_jobs examples/sessions/attention_early_merge.py --gpus 0 --logdir /absolute/new/logs
A retry uses the SAME C6_WORK. Its deadline does not move.
"""
MAX_SESSION_SECONDS=2400
BASE='python -u -m examples.vllm_llm.attention_early_merge.session --work "${C6_WORK:?set absolute C6_WORK}" --model "${C6_MODEL:?set packed C6_MODEL}"'
JOBS=[dict(name='c6_build',lane='cpu',cmd=BASE+' build',timeout=1800,quiet_s=60),
      dict(name='c6_measure',lane='gpu',gpu=0,reads=['c6_build'],cmd=BASE+' run',timeout=1800,quiet_s=60)]
STUDIES={'C6 attention combined protocol':['c6_measure']}
