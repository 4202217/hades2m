// The host symbols the graphics layer uses but the unit tests do not provide.
// The unit tests have no guest address space: the guest heap range is empty and the big-stack trampoline is never reached.
#include <stdint.h>
#include <stddef.h>
#include <stdatomic.h>
int dm_guest_heap_range(uint8_t **base, size_t *size) { (void)base; (void)size; return 0; }
int dm_guest_renewable(uint64_t a, size_t n) { (void)a; (void)n; return 0; }   // no guest memory: the D1 pool page renew never runs in the unit tests
void dm_bigstack_enter(void) {}
// The serial numbers for the texture / program objects live in hle_nvn.c, which some tests exclude from the link.
__attribute__((weak)) uint64_t nvn_new_serial(void) { static _Atomic uint64_t n; return ++n; }
// The front end's symbols: the unit tests have no window, so there is no Documents folder and the picture fills the frame
// (no bars). front_metal_layer is stubbed per test (t_nvn.c has its own), front_set_picture_rect has a weak default in the
// graphics layer, and the rest live in the front end, which the tests do not link.
const char *front_documents_dir(void) { return NULL; }
uint64_t front_game_res(int docked) { return docked ? (900ull << 32 | 1600) : (720ull << 32 | 1280); }
