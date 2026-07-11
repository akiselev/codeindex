(
  (method_declaration
    name: (identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "method")
)

(
  (constructor_declaration
    name: (identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "constructor")
)

(
  (local_function_statement
    name: (identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (lambda_expression
    body: (block) @unit.body) @unit
  (#set! unit.kind "lambda")
)
