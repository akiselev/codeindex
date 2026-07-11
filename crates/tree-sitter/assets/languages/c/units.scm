(
  (function_definition
    declarator: (function_declarator
      declarator: (identifier) @unit.name)
    body: (compound_statement) @unit.body) @unit
  (#set! unit.kind "function")
)
