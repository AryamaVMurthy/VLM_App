#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_KV_CACHE_INTERFACE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_KV_CACHE_INTERFACE_H_

#include <cstddef>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// The KV cache interface including all K and V buffers for a model.
class KVCacheInterface {
 public:
  virtual ~KVCacheInterface() = default;

  // Resizes the KV cache to the specified number of entries.
  // Note: If the requested `num_entries` is smaller than the current number
  // of entries, the cache will be trimmed to the requested size.
  virtual absl::Status Resize(size_t num_entries) = 0;

  // Returns the total number of entries in the KV cache per block.
  virtual int GetNumEntries() const = 0;

  // Serializes the KV cache to a byte string.
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Loads the KV cache from a serialized byte string.
  virtual absl::Status Load(absl::string_view serialized_kv_cache) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_KV_CACHE_INTERFACE_H_
