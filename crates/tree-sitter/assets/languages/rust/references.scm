; reference/call-site captures for the Usage channel (@ref.callee).
; Function and method calls: capture the callee expression (`foo`,
; `Type::foo`, `expr.method`); the resolver reduces it to a symbol name.
(call_expression
  function: (_) @ref.callee)

; Macro invocations: `foo!( ... )`.
(macro_invocation
  macro: (identifier) @ref.callee)
