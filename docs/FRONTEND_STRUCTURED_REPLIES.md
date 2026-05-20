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

---

## `type`: `order_suggestions` (display only — not `assistant_structured`)

Sent on the **same WebSocket** after a voice `place_order`, usually just before `"type": "done"`. **No TTS.** Guest still orders by voice.

```json
{
  "type": "order_suggestions",
  "turn_id": 1,
  "order_id": "<mongo order id or null>",
  "payload": {
    "title": "Pairs well with Paneer Tikka",
    "triggered_by": [{ "dish_id": 101, "name": "Paneer Tikka" }],
    "items": [
      {
        "dish_id": 12,
        "name": "Mango Lassi",
        "price": 4.5,
        "image": "https://...",
        "info": "beverage · ..."
      }
    ]
  }
}
```

Replace the suggestion strip when a new message arrives (same turn may only send one). Ordering is unchanged — do not call a separate add-to-cart API for taps in v1.
