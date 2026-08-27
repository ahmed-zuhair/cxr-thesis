# Objective 3 v1.1 enhancement protocol amendment

- Original v1.0 recorded SHA-256: `3fa2199e3188a3c0d9c1fc29a7e27c0510965c73524982d5f9a7094c8d796944`
- v1.1 amendment SHA-256: `493fb4e3fe45e874f09635f96d7d423e39a081a3579d10343bb487f64c2eb4e6`
- Architecture: `v1_1_reupload_gated`
- Seeds: 42, 43 and 44
- Quantum bottleneck parameters: 36
- Classical bottleneck parameters: 36
- Total trainable parameters per head: 3,253
- Test cohort selected: no
- Test manifest opened: no
- Test labels accessed: no
- Test evaluated: no

The original v1.0 JSON was stored only in Kaggle's ephemeral
`/kaggle/working` directory and was lost when the runtime restarted.
Its SHA-256 and successful test-blind lock output remain recorded in
the notebook. This amendment explicitly records that the missing file
could not be byte-for-byte re-verified.

No patient identifiers, image identifiers, medical images, masks,
case-level predictions or private checkpoints are included.
