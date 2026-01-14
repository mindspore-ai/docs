# Struct Key

```cpp
using Key = struct MS_API Key {
  size_t max_key_len = 32;
  size_t len = 0;
  unsigned char key[32] = {0};
  Key() : len(0) {}
  explicit Key(const char *dec_key, size_t key_len)
};
```

键结构体。
