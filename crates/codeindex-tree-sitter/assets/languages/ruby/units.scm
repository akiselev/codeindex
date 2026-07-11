(
  (method
    name: (_) @unit.name
    body: (body_statement) @unit.body) @unit
  (#set! unit.kind "method")
)

(
  (singleton_method
    name: (_) @unit.name
    body: (body_statement) @unit.body) @unit
  (#set! unit.kind "method")
)

(
  (lambda
    body: (_) @unit.body) @unit
  (#set! unit.kind "closure")
)
