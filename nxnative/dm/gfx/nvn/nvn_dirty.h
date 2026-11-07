// Write-protect dirty-page tracking for guest memory (nvn_dirty.c)
#pragma once
#include <stddef.h>
#include <stdint.h>

void nvn_dirty_add_range(const void *mem, size_t size);           // Register a memory pool as a tracked range (the first call registers the page-fault hook and the host pre-write hook)
void nvn_dirty_stat(int *live, int *peak, int *cap);                // number of registered memory pools, peak, tracking table capacity (for the memory readout line)
void nvn_dirty_remove_range(const void *mem);                     // Pool destruction: restore protected pages to writable and mark invalid
int nvn_dirty_tracked(const void *addr, size_t len);
void nvn_dirty_protect(const void *addr, size_t len);              // Set the covered pages read-only
uint64_t nvn_dirty_now(void);                                      // Current generation: taken before upload, as the stamp of that content
int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp); // Covered pages were written after the stamp -> 1
int nvn_dirty_span_since(const void *addr, size_t len, uint64_t stamp, uint64_t *pages, uint64_t *first_off, uint64_t *last_end); // Diagnostic: dirty page count and byte span since stamp; -1 = untracked
void nvn_dirty_stats(uint64_t *faults, uint64_t *preps);           // Cumulative: number of page faults triggered by game writes, number of pages unprotected before host writes
#define NVN_DIRTY_MAX_PAGES 4096
int nvn_dirty_content_hashes(const void *addr, size_t len, uint64_t stamp, uint64_t *out, int max_pages, uint64_t *first_off, uint64_t *last_end);   // The hashes of the pages written after the stamp (the dirty ones only). -1 = untracked.
int nvn_dirty_all_hashes(const void *addr, size_t len, uint64_t *out, int max_pages);   // The hashes of every page of the range, in page order. -1 = untracked.
void nvn_dirty_writer_counts(uint64_t *packaged, uint64_t *texels, uint64_t *fs);   // DM_NVN_DIRTYWRITER: the host's own writes into pool pages, by path
void nvn_dirty_mp_stats(uint64_t *calls, uint64_t *pages, uint64_t *ns);   // Cumulative: this module's mprotect call count, pages covered, time spent
void nvn_dirty_slot_stats(int *live, int *used_slots, uint64_t *adds_total); // Currently live slots, slot table high-water mark, total number of registrations
