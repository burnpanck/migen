
from ..structure import (
    Signal, ClockSignal, ResetSignal,
    _Operator,
    If,
    _Slice, Cat, Replicate,
)
from .explicit_migen_types import *
from .type_annotator import ExplicitTyper
from .types import *
from .syntax import Verilog2VHDL_operator_map
from .ast import (
    Boolean,
    TestIfNonzero,
    VHDLSignal,
    EntityBody,
    NodeTransformer,
)

class VHDLReprGenerator:
    def __init__(self,overrides={},*,single_bit=std_logic,unsigned=unsigned,signed=signed,boolean=boolean):
        self.overrides = overrides
        self.boolean = boolean
        self.single_bit = single_bit
        self.unsigned = unsigned
        self.signed = signed
        self.typer = ExplicitTyper(raise_on_type_mismatch=True)

    def VHDL_representation_of_Signal(self,signal):
        if not isinstance(signal,AbstractTypedExpression):
            signal = self.typer.visit(signal)
#        assert isinstance(signal,)
        migen_type = signal.type
        return migen_type, self.VHDL_representation_for(migen_type)

    def VHDL_representation_for(self, migen_type):
        """ Given a migen type, find a suitable VHDL representation.

        If argument already has a specified VHDL representation, returns that one.
        """
        assert isinstance(migen_type,MigenExpressionType)
        if isinstance(migen_type,Boolean):
            return self.boolean
        if not isinstance(migen_type,MigenInteger):
            raise TypeError("Don't know how to map type %s to VHDL"%migen_type)
        if isinstance(migen_type, Unsigned) and migen_type.nbits == 1:
            # Verilog has no boolean type, so is this a 1-element array or a single wire?
            if self.single_bit is not None:
                return self.single_bit
        typ = self.signed if isinstance(migen_type, Signed) else self.unsigned
        return typ[migen_type.nbits-1:0]

natural_repr = VHDLReprGenerator(single_bit=std_logic,unsigned=unsigned,signed=signed,boolean=boolean)
only_numeric = VHDLReprGenerator(single_bit=None,unsigned=unsigned,signed=signed,boolean=unsigned[0:0])
all_slv = VHDLReprGenerator(single_bit=std_logic,unsigned=std_logic_vector,signed=std_logic_vector,boolean=std_logic)

class ToVHDLLowerer(NodeTransformer):
    """ Converts a Migen AST into an AST suitable for VHDL export

    In this process, it assigns a suitable VHDL type to any expression.
    """

    def __init__(self,ns,*,vhdl_repr=natural_repr,replaced_signals={}):
        # configuration (constants)
        self.vhdl_repr = vhdl_repr

        # global variables
        self.ns = ns
        self.replaced_signals = replaced_signals

        # context variables
        self.entity = None  # the entity body we're currently building

    @context_for(EntityBody)
    def EntityBody_ctxt(self, node):
        return dict(
            entity = node,
        )

    def assign_VHDL_repr(self, node):
        if node.repr is None:
            node.repr = self.vhdl_repr.VHDL_representation_for(node.type)

    @visitor_for(AbstractTypedExpression)
    def visit_Typed(self, node):
        self.assign_VHDL_repr(node)
        return node

    @visitor_for(MigenExpression,needs_original_node=True)
    def visit_Annotated(self, orig, node):
        # nested nodes have already been processed, now generate an appropriate VHDL type for this node
        self.assign_VHDL_repr(node)
        # now delegate to visitors
        return super().visit_Annotated(orig,node)

    @visitor_for_wrapped(Signal, ClockSignal, ResetSignal)
    def wrapped_Signal(self, node):
        sig = node.expr
        repl = self.replaced_signals.get(sig,None)
        if repl is not None:
            # this Signal already has a replacement.
            # TODO: should we again check if the type in this wrapper conforms to the replacement signal?
            return repl
        # this Signal does not yet have a replacement.
        ret = VHDLSignal(
            name = self.ns.get_name(sig),
            type = node.type,
            repr = self.vhdl_repr.VHDL_representation_for(node.type),
            reset = self.visit(sig.reset) if sig.reset is not None else None,
        )
        # add to replacement map
        self.replaced_signals[sig] = ret
        # add signal to entity
        assert not ret.name in self.entity.entity.ports
        assert not ret.name in self.entity.signals
        self.entity.signals[ret.name] = ret
        return ret

    @visitor_for_wrapped(_Operator)
    def visit_Operator(self, node):
        expr = node.expr
        repr = node.repr
        op = Verilog2VHDL_operator_map[expr.op]  # TODO: op=='m'
        if op in {
            'and', 'or', 'nand', 'nor', 'xor', 'xnor', # logical operators
            'not',                                     # unary logical operators
            '+', '-',                                  # addition operators (two operands)
            '<', '<=', '=', '/=', '>', '>=',           # relational operators
            '+', '-',                                  # unary operators (one operand)
        }:
            assert all(
                isinstance(v.type, MigenInteger)
                for v in expr.operands
            ) # only Migen semantics are implemented
            # VHDL and migen semantics match as long as the input types match
            # and are either signed or unsigned
            migen_repr = only_numeric.VHDL_representation_for(node.type)
            expr.operands = [
                TypeChange.if_needed(v,repr=migen_repr)
                for v in
                expr.operands
            ]
            return TypeChange.if_needed(node,repr=repr)
        elif op in {'sll', 'srl', 'sla', 'sra', 'rol', 'ror'}:
            # shift operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        elif op in {'&'}:
            # concatenation operator (same precedence as addition operators)
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        elif op in {'*', '/', 'mod', 'rem'}:
            # multiplying operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        elif op in {'**', 'abs'}:
            # misc operators
            raise NotImplementedError(type(self).__name__ + '.visit_Operator: operator ' + op)
        else:
            raise TypeError('Unknown operator "%s" with %d operands' % (node.op, len(node.operands)))

    @visitor_for_wrapped(_Slice)
    def visit_Slice(self, wrapnode):
        node = wrapnode.expr
        if node.stop - node.start == 1:
            # this is not a slice, but an indexing operation!
            wrapnode.repr = node.value.repr.valuetype
        else:
            wrapnode.repr = node.value.repr.ultimate_base[node.stop-1:node.start]
        return wrapnode

    @visitor_for_wrapped(Cat)
    def visit_Cat(self, wrapnode):
        node = wrapnode.expr
        nbits = 0
        for o in node.l:
            type = o.repr
            if not isinstance(type, VHDLArray):
                nbits += 1
            else:
                l = type.indextypes[0].length
                nbits += l
        wrapnode.repr = unsigned[nbits - 1:0]
        return wrapnode

    @visitor_for(If)
    def visit_If(self, node):
        # make implicit test explicit
        if not isinstance(node.cond.type,Boolean):
            node.cond = TestIfNonzero(node.cond)
        return node

    @visitor_for(TestIfNonzero)
    def visit_TestIfNonzero(self,node):
        node.expr = TypeChange.if_needed(node.expr,repr=integer)
        repr = node.repr
        node.repr = boolean
        return TypeChange.if_needed(node,repr=repr)