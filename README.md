# simd
simd intrinsics header library

Compile with `-march=native`

There will be many warnings about ignored attributes, this is due to the `__m*` types containing attributes and being put into template arguments. This could be removed by detemplatising everything, but that would result in a lot more code.

Not all operations are supported on the various vector types, e.g., multiplication on vectors of int sized 8, 16 and 64 - looking to provide workarounds for this, see [^MultWorkaround]

Current support operations are '+', '-', '*', '/'. There is intention to add further operations, see [^AdditionalOperations]

---

- [x] Get "something" working
- [ ] Complete basic ops for all types
- [ ] Include 512bit types
- [ ] Create a means for operations to happen on large arrays - auto load and loop on interations

---

[^MultWorkaround]: Possible multiply operation workarounds include:
    - Revert back to using non-intrinsic methods
    - Use alternative intrinsics to effectively perform a multiply
    - Upcast/downcast to support multiplication, e.g. convert vec16_int8_t to vec16_int16_t, or vec4_int64_t to vec4_int32_t
    - A combination of the above...

[^AdditionalOperations]: Additional operations to add:
    - sum(reduce)
    - absolute
    - multiply add
    - clamp
    - and
    - or
    - min
    - max...