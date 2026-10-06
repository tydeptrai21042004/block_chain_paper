# Baseline fidelity rules

Numerical rows are separated by what is actually reproduced.

| Baseline | Executable condition | Fidelity label |
|---|---|---|
| RDoC | ordered trace + atomic `one-step` terminal action | structural projection |
| TrueBit | ordered trace + atomic `one-step` terminal action | structural projection |
| Arbitrum | ordered trace + atomic `one-step` terminal action | structural projection |
| opML single-phase | trace granularity is VM microinstruction + `one-step` terminal action | faithful localization level |
| opML outer phase | ordered operator/high-level trace | outer-phase projection only |
| Agatha GPP | ordered chain only | chain projection only |
| Kirkpatrick-Klawe | forced atomic leaves + constant finite query cost | exact objective reduction |
| Hu-Tucker | forced atomic leaves + constant finite query cost | exact objective reduction |
| Height-limited alphabetic | forced atomic leaves + constant finite query cost + height bound | exact objective reduction |
| zk-OPML | atomic operator has explicit `zk-proof`-capable measured terminal cost | operator/ZK-terminal projection |

The artifact never inserts a native-replay number into a missing ZK row and never labels an operator-only opML projection as a full two-phase opML reproduction.
