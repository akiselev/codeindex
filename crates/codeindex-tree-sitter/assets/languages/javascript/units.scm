(
  (function_declaration
    name: (identifier) @unit.name
    body: (statement_block) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (method_definition
    name: (_) @unit.name
    body: (statement_block) @unit.body) @unit
  (#set! unit.kind "method")
)

(
  (arrow_function
    body: (_) @unit.body) @unit
  (#set! unit.kind "closure")
)

(
  (function_expression
    body: (statement_block) @unit.body) @unit
  (#set! unit.kind "closure")
)
