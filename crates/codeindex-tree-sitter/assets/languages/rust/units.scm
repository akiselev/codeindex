(
  (function_item
    name: (identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (closure_expression
    body: (_) @unit.body) @unit
  (#set! unit.kind "closure")
)
