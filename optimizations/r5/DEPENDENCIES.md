# Pinned runtime dependencies

The standalone R5 build reuses these two existing dependency binaries.
Paths are installation-specific; hashes identify the exact measured libraries.

| Library | SHA256 |
| --- | --- |
| `libinfinicore_cpp_api.so` | `27fa70495fc38bde729eee8f4833f1812fc895c5776e11bfb7a1c81c2dd4f76a` |
| `libinfiniop.so` | `7dc2b1dcbdaefef794899fa8839d7826ca9e46ad2b9b59855b83854be6683d85` |

The InfiniCore binary has logging-symbol isolation; the operator binary contains the R5
RoPE backend change. `src/rope_nvidia.cu`, `rope_backend_policy.hpp` and
`rope_backend_runtime.hpp` preserve the backend implementation for transfer into the
matching InfiniCore source tree. `src/logging.map` preserves the export policy.

These files do not constitute a clean rebuild of all InfiniCore dependencies. A different
dependency build requires its own validation and must not inherit the recorded performance
claim merely by using the same source revision. SDK/system dependencies remain external.
