(
  (method_declaration
    name: (identifier) @unit.name
    body: (block) @unit.body) @unit
  (#set! unit.kind "method")
)

(
  (constructor_declaration
    name: (identifier) @unit.name
    body: (constructor_body) @unit.body) @unit
  (#set! unit.kind "constructor")
)

(
  (lambda_expression
    body: (_) @unit.body) @unit
  (#set! unit.kind "lambda")
)
