use codeindex_core::EntityKind;

#[test]
fn frontend_specific_kind_spellings_remain_lossless() {
    for value in ["lambda", "class", "struct", "enum", "const"] {
        let kind = EntityKind::from(value);
        assert_eq!(kind.as_str(), value);
    }
}

#[test]
fn canonical_kinds_use_typed_variants() {
    assert_eq!(EntityKind::from("function"), EntityKind::Function);
    assert_eq!(EntityKind::from("closure"), EntityKind::Closure);
    assert_eq!(EntityKind::from("constant"), EntityKind::Constant);
}
