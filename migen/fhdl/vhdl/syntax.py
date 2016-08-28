


reserved_keywords = {
    "abs", "access", "after", "alias", "all", "and", "architecture", "array", "assert",
    "attribute", "begin", "block", "body", "buffer", "bus", "case", "component",
    "configuration", "constant", "disconnect", "downto", "else", "elsif", "end",
    "entity", "exit", "file", "for", "function", "generate", "generic", "group",
    "guarded", "if", "impure", "in", "inertial", "inout", "is", "label", "library",
    "linkage", "literal", "loop", "map", "mod", "nand", "new", "next", "nor", "not",
    "null", "of", "on", "open", "or", "others", "out", "package", "port", "postponed",
    "procedure", "process", "pure", "range", "record", "register", "reject", "return",
    "rol", "ror", "select", "severity", "shared", "signal", "sla", "sli", "sra", "srl",
    "subtype", "then", "to", "transport", "type", "unaffected", "units", "until"
}


Verilog2VHDL_operator_map = {v[0]:v[-1] for v in (v.split(':') for v in '''
 &:and |:or ^:xor
 < <= ==:= !=:/= > >=
 <<:sll >>:srl <<<:sla >>>:sra
 + -
 *
 ~:not
'''.split())}