Verilog2VHDL_operator_map = {v[0]:v[-1] for v in (v.split(':') for v in '''
 &:and |:or ^:xor
 < <= ==:= !=:/= > >=
 <<:sll >>:srl <<<:sla >>>:sra
 + -
 *
 ~:not
'''.split())}


literal_printers = {
    integer: lambda v, l: str(v),
    unsigned: lambda v, l: '"' + bin(v)[2:].rjust(l, '0') + '"',
    signed: lambda v, l: '"' + bin(v & ~-(1 << l))[2:].rjust(l, '0') + '"',
    std_logic: lambda v, l: "'1'" if v else "'0'",
    boolean: lambda v, l: "true" if v else "false",
}

# ---------------------------------
# integer representation converters
#
# for all integers simultaneously representable in both representations
# the conversion should be one to one. Otherwise the result is undefined.
def conv(template):
    def convert(repr, expr, orig=None):
        return template.format(
            x=expr,
            l=repr.indextypes[0].length if isinstance(repr, VHDLArray) else None,
        )
    return convert
integer_repr_converters = {
    (integer,std_logic):conv('to_integer(to_unsigned({x},1))'),   # '1', 'H' -> 1, else 0
    (integer,signed):conv('to_integer({x})'),   # one-to-one
    (integer,unsigned):conv('to_integer({x})'), # one-to-one
    (signed,integer):conv('to_signed({x},{l})'), # ?
    (unsigned,std_logic):conv('to_unsigned({x},{l})'),   # '1','H' -> 1, else -> 0
    (unsigned,integer):conv('to_unsigned({x},{l})'),   # ?
    (std_logic,boolean):conv("to_std_ulogic({x})"), # '1','H' -> true, else -> false
    (integer,boolean):conv("to_integer(to_unsigned(to_std_ulogic({x}),1))"), # true -> '1', false -> '0'
    (std_logic,integer):conv("get_index(to_unsigned({x},1),0)"), # truncates
    (std_logic,unsigned):conv("get_index({x},0)"), # truncates
#    (boolean,std_logic):conv("({x} = '1')"),
    (boolean,integer):conv('({x} /= 0)'),
    (boolean,unsigned):conv("(to_integer({x}) /= 0)"), # 0 -> false, else -> true
}


class VHDLPrinter(VHDLNodeTransformer):
    """ Write actual VHDL from an AST prepared with :py:class:`ToVHDLConverter`.

    Note that not all AST's are directly representable in VHDL.
    In addition to that, this converter makes some assumptions on the representation types,
    which are not necessarily checked here, if they are ensured by :py:class:`ToVHDLConverter`.
    Thus, :py:class:`VHDLPrinter` should always be used in conjunction with :py:class:`ToVHDLConverter`.

    """

    def __init__(self):
        # configuration (constants)

        # global variables

        # context variables
        pass


    def recurse_unknown_wrapped_node(self,node):
        # do not let self.visit for the wrapped node try to re-assemble a node
        # instead, simply jump over
        return self.visit(node.expr)


    @visitor_for_wrapped(Constant)
    def visit_Constant(self, node):
        printer = literal_printers[node.repr]
        return printer(node.expr.value)

    @combiner_for(TypeChange)
    def combine_TypeChange(self, node, expr, *, type=None, repr=None):
        return expr

    @visitor_for(TypeChange, needs_original_node=True)
    def visit_TypeChange(self, orig, expr):
        # TypeChange conversions are supposed to keep the represented value intact, thus they
        # depend on the interpretation of the representation.
        if not isinstance(orig.type, MigenInteger) or not isinstance(orig.expr.type, MigenInteger):
            # so far, only conversions between MigenInteger is supported
            raise TypeError('Undefined (representation) conversion between types %s and %s' % (orig.expr.type, orig.type))
        conv = integer_repr_converters.get((orig.repr.ultimate_base, orig.expr.repr.ultimate_base))
        if conv is None:
            raise TypeError('Unknown conversion between integer representations %s and %s' % (orig.expr.repr, orig.repr))
        return conv(orig.repr, expr, orig.expr.repr)

    @visitor_for(TestIfNonzero, needs_original_node=True)
    def visit_TestIfNonzero(self, orig, expr):
        if not (
            (orig.expr.repr.ultimate_base == integer)
            and (orig.repr == bool)
        ):
            raise TypeError('VHDL TestIfNonzero works only on integer and returns boolean')
        return '('+expr + ' /= 0)'

    @visitor_for_wrapped(Signal,ClockSignal,ResetSignal)
    def visit_Signal(self, node):
        raise TypeError("Bare MyHDL signals should be replaced with VHDLSignals first: "+str(node.expr))

    @combiner_for_wrapped(_Operator)
    def combine_Operator(self, wrapnode, op, operands):
        node = wrapnode.expr


        migen_bits, migen_signed = value_bits_sign(node)
        op = Verilog2VHDL_operator_map[op]  # TODO: op=='m'
        if op in {'and','or','nand','nor','xor','xnor'}:
            # logical operators
            left,right = operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,type)
            return '('+lex + ' ' + op + ' ' + rex+')', type
        elif op in {'<','<=','=','/=','>','>='}:
            # relational operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,type)
            return '('+lex + op + rex+')', boolean
        elif op in {'sll','srl','sla','sra','rol','ror'}:
            # shift operators
            left,right = node.operands
            lex,type = self.visit(left)
            rex = self.visit_as_type(right,integer)
            return '('+lex + ' ' + op + ' ' + rex+')', type
        elif op in {'+','-'} and len(node.operands)==2:
            # addition operators
            left,right = node.operands
            if False:
                # VHDL semantics
                lex,type = self.visit(left)
                rex = self.visit_as_type(right,type)
            else:
                # emulate Verilog semantics in VHDL
                type = (signed if migen_signed else unsigned)[migen_bits-1:0]
                lex = self.visit_as_type(left,type)
                rex = self.visit_as_type(right,type)
            return '('+lex + op + rex+')', type
        elif op in {'&'}:
            # concatenation operator (same precedence as addition operators)
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        elif op in {'+','-'} and len(node.operands)==1:
            # unary operators
            right, = node.operands
            ex,type = self.visit(right)
            return '(' + op + ex+')', type
        elif op in {'*','/','mod','rem'}:
            # multiplying operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        elif op in {'not'}:
            right, = node.operands
            ex,type = self.visit(right)
            return '(' + op + ' ' + ex+')', type
        elif op in {'**','abs'}:
            # misc operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator '+op)
        else:
            raise TypeError('Unknown operator "%s" with %d operands'%(node.op,len(node.operands)))

    def visit_Slice(self, node):
        expr,type = self.visit(node.value)
        if not isinstance(type,VHDLArray):
            raise TypeError('Cannot slice value of non-array type %s'%type)
        idx, = type.indextypes
        if node.stop - node.start == 1:
            # this is not a slice, but an indexing operation!
            return expr + '(' + str(node.start) + ')', type.valuetype
        return expr + '(' + str(node.stop-1) + ' downto ' + str(node.start) + ')', type.ultimate_base[node.stop-1:node.start]

    def visit_Cat(self, node):
        pieces = []
        nbits = 0
        for o in node.l:
            expr,type = self.visit(o)
            if not isinstance(type,VHDLArray):
                pieces.append(self._convert_type(std_logic,expr,type))
 #               pieces.append(expr)
                nbits += 1
            else:
                l = type.indextypes[0].length
                pieces.append(self._convert_type(unsigned[l-1:0],expr,type))
#                pieces.append(expr)
                nbits += l
        expr = "unsigned'(" + '&'.join(reversed(pieces)) + ')';
        return expr, unsigned[nbits-1:0]

    def visit_Replicate(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_Replicate')

    def visit_Assign(self, node):
        return self._cannot_visit(node)

    def visit_If(self, node):
        return self._cannot_visit(node)

    def visit_Case(self, node):
        return self._cannot_visit(node)

    def visit_Fragment(self, node):
        return self._cannot_visit(node)

    def visit_statements(self, node):
        return self._cannot_visit(node)

    def visit_clock_domains(self, node):
        return self._cannot_visit(node)

    def visit_ArrayProxy(self, node):
        raise NotImplementedError(type(self).__name__+'.visit_ArrayProxy')
        return _ArrayProxy([self.visit(choice) for choice in node.choices],
            self.visit(node.key))

    def visit_unknown(self, node):
        return self._cannot_visit(node)

    def _cannot_visit(self, node):
        raise TypeError('Node of type "%s" cannot be written as a VHDL expression'%type(node).__name__)