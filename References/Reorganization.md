# Project reorganization on November 2

The existing repository and its Git history moved under 01_Assignments intact.
Phase I and II assignment materials were consolidated into it. Superseded Phase II
support remains local under Archive/Part_II_Previous_Workspace. No datasets or model
binaries were deleted or uploaded.

| Before | Authoritative location now |
|---|---|
| Repository beside 01_Assignments | 01_Assignments/CAI2820C_FraudShield |
| Part_II_Data_Preparation and assignment Part II | Part_II |
| Part_III_Modeling_Demo training/evaluation | Part_III/Modeling |
| Part_III_Modeling_Demo app/static | Final_Demo |
| Common feature/scoring modules | Shared |
| data | Data |
| artifacts | Models |
| results | Part_III/Results |
| Root final forms and rubrics | Part_III/Deliverables and Part_III/References |

Old executed outputs and metadata may show original absolute paths. They are
historical records, not current run instructions. Current entry points discover
the root through project_paths.py. Original training-source manifest fingerprints
are retained beside the migrated manifests. Checks on all three saved models'
example predictions pass. No 2026 test was rerun or model retrained.

Windows may retain empty old directories while they are open in another program.
Those contain no authoritative files and must not be used as a second project.
