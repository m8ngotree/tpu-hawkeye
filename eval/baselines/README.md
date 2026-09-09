# Baseline runs

Same JAXBench workloads, same harness, same tools, same turn budget -- except
`read_taxonomy_cell`/`list_taxonomy_cells` are removed from the agent's toolset (or
swapped for something like raw Pallas/Mosaic docs, matching Hawkeye's baseline
sweep in Table 4). This is the number the taxonomy-grounded run needs to beat;
without it "our agent gets N% MXU utilization" is meaningless.

Store outputs here as `<model>_<date>.json`, matching the schema
`eval/run_agent_eval.py` writes.
