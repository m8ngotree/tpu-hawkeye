# Baseline runs

Same JAXBench workloads, same loop, `use_taxonomy=False` -- i.e. the LLM proposes
patches grounded only in whatever docs/RAG context you give it, not the taxonomy
cells. This is the number the taxonomy-grounded run needs to beat; without it
"our agent gets N% MXU utilization" is meaningless.

Store outputs here as `<model>_<date>.json`, matching the schema
`eval/run_agent_eval.py` writes.
