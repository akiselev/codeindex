(
  (function_definition
    name: (name) @unit.name
    body: (compound_statement) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (method_declaration
    name: (name) @unit.name
    body: (compound_statement) @unit.body) @unit
  (#set! unit.kind "method")
)

(
  (anonymous_function
    body: (compound_statement) @unit.body) @unit
  (#set! unit.kind "closure")
)
