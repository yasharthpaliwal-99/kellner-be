# WebSocket: structured assistant replies

New type introduced- `"type": "assistant_structured"` → this message is the structured reply. The body has `payload` (object) and `mode`. The frontend handles it in the branch where `type === "assistant_structured"`.

`"type": "assistant_text_delta"` → this message is the normal assistant text path (streaming chunks on regular turns, or sometimes a JSON string on structured turns in addition to `assistant_structured`).

---

## `payload` shapes (`assistant_structured`)

### `mode`: `bill`

```json
{
  "items": [
    { "name": "string", "quantity": 1, "price": 12.5 }
  ],
  "total": 25.0
}
```

Use `null` for `price` or `total` when unknown.

### `mode`: `order_confirmation`

```json
{
  "items": [
    { "name": "string", "quantity": 1 }
  ]
}
```

### `mode`: `recommendations`

```json
{
  "recommendation_focus": "string",
  "items": [
    { "name": "string", "quantity": 1, "price": 12.5 }
  ]
}
```

Use `null` for `price` when unknown.
