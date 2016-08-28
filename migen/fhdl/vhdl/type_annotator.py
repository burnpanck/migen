
from ..structure import (
    Constant, Signal, ClockSignal, ResetSignal,
    _Operator, _Slice, _Value, _ArrayProxy,
    Cat, Replicate
)
from ..visit_generic import visitor_for
from .explicit_migen_types import NodeTransformer, Signed, Unsigned, MigenExpression, AbstractTypedExpression

class ExplicitTyper(NodeTransformer):
    """ Adds explicit type annotations to untyped Migen nodes.

    """
    def __init__(self,*,raise_on_type_mismatch=True):
        """
        raise_on_type_mismatch: If pre-existing type annotations do not match, raise an exception rather than just replacing
        the annotation.
        """
        self.raise_on_type_mismatch = raise_on_type_mismatch

    def type_wrap(self,node,type):
        """ Annotate node with type.
        """
        return MigenExpression(node,type)

    @visitor_for(AbstractTypedExpression)
    def visit_AlreadyTyped(self,node):
        """ Leave alone... this transformer is only concerned with nodes that are not inherently typed. """
        return node

    @visitor_for(MigenExpression,needs_original_node=True) # overrides visit_AlreadyTyped due to beeing more specific
    def visit_PreviousAnnotation(self,orig,node):
        ret = node.expr # we'll remove outer layer of annotation, which is the previous annotation
        assert isinstance(ret,MigenExpression)
        if self.raise_on_type_mismatch and not ret.type==orig.type:
            raise TypeError('Mismatching type annotations found for expression %s: %s != %s'%(ret.expr,ret.type,orig.type))
        return node

    @visitor_for(Constant)
    def visit_Constant(self, node):
        return self.type_wrap(node,(Signed if node.signed else Unsigned)(node.nbits,min=node.value,max=node.value))

    @visitor_for(Signal)
    def visit_Signal(self, node):
        return self.type_wrap(node,(Signed if node.signed else Unsigned)(node.nbits))

    @visitor_for(ClockSignal,ResetSignal)
    def visit_BoolSignal(self, node):
        return self.type_wrap(node,Unsigned(1,min=0,max=1))

    @visitor_for(_Operator)
    def visit_Operator(self, node):
        from ..bitcontainer import operator_bits_sign
        obs = [(n.type.nbits,isinstance(n.type,Signed)) for n in node.operands]
        nbits, signed = operator_bits_sign(node.op, obs)
        typ = (Signed if signed else Unsigned)(nbits)
        return self.type_wrap(node,typ)

    @visitor_for(_Slice)
    def visit_Slice(self, node):
        typ = (Signed if isinstance(node.value.type,Signed) else Unsigned)(node.stop - node.start)
        return self.type_wrap(node,typ)

    @visitor_for(Cat)
    def visit_Cat(self, node):
        typ = Unsigned(sum(sv.type.nbits for sv in node.l))
        return self.type_wrap(node,typ)

    @visitor_for(Replicate)
    def visit_Replicate(self, node):
        typ = Unsigned(node.v.type.nbits*node.n)
        return self.type_wrap(node,typ)

    @visitor_for(_ArrayProxy)
    def visit_ArrayProxy(self, node):
        typ = (
            Signed if
            any(isinstance(n.type,Signed) for n in node.choices)
            else Unsigned
        )(max(n.type.nbits for n in node.choices))
        return self.type_wrap(node,typ)

    @visitor_for(_Value)
    def visit_other_Expression(self, node):
        # catch all for _Value, assume every expression is a subclass of _Value
        raise TypeError("Don't know how to generate type for expression node %s"%node)
