# simd
simd intrinsics header library

Compile with `-march=native`

A naive vector class has been implemented which loops through the elements of the vector performing operations through simd instructions. This could be inefficient as each operation  performed requires an additional loop on the data. It would be worth investigating a means store a number of intended operations to perform then do so in a single loop.

### example usage

<pre><code>
  simd::vector<std::int16, 20> a = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  simd::vector<std::int16, 20> b = a + a;
  
  for (auto i : b.data)
  {
      std::cout << i << ' ';
  }
</code></pre>

outputs:

2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40

### additional info

Not all operations are supported on the various vector types, e.g., multiplication and division on various sized integer vectors - looking to provide workarounds for this, see [^MultDivWorkaround]

Current supported operations are '+', '-', '*', '/'. There is intention to add further operations, see [^AdditionalOperations]

---

- [x] Get "something" working
- [ ] Complete basic ops for all types
- [ ] Include 512bit types
- [x] Create a means for operations to happen on large arrays - auto load and loop on interations
- [ ] Create a matrix class

---

[^MultDivWorkaround]: Possible multiply operation workarounds include:
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
    - max
    - shift
    - binary comparison operations