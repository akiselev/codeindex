(
  (function_definition
    declarator: (function_declarator
      declarator: [
        (identifier) @unit.name
        (field_identifier) @unit.name
        (qualified_identifier
          name: [
            (identifier) @unit.name
            (field_identifier) @unit.name
          ])
      ])
    body: (compound_statement) @unit.body) @unit
  (#set! unit.kind "function")
)

(
  (lambda_expression
    body: (compound_statement) @unit.body) @unit
  (#set! unit.kind "closure")
)
