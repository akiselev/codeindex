(
  (function_declaration
    name: (identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (method_declaration
    name: (field_identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "method")
)

(
  (func_literal
    body: (block) @unit.body) @unit
  (#set! unit.kind "closure")
)
