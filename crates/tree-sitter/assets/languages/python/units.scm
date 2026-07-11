(
  (function_definition
    name: (identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (lambda
    body: (_) @unit.body) @unit
  (#set! unit.kind "lambda")
)
