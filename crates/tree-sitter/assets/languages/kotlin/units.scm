(
  (function_declaration
    name: (identifier) @unit.name
    (function_body) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (secondary_constructor
    (block) @unit.body) @unit
  (#set! unit.kind "constructor")
)

(
  (lambda_literal) @unit @unit.body
  (#set! unit.kind "lambda")
)

(
  (anonymous_function
    (function_body) @unit.body) @unit
  (#set! unit.kind "closure")
)
